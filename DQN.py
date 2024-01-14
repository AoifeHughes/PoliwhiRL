# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from itertools import count
import os
from tqdm import tqdm
from PIL import Image
import csv
from controls import Controller


class DQN(nn.Module):
    def __init__(self, h, w, outputs, USE_GRAYSCALE):
        super(DQN, self).__init__()
        self.USE_GRAYSCALE = USE_GRAYSCALE
        self.conv1 = nn.Conv2d(1 if USE_GRAYSCALE else 3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self._to_linear = None
        self._compute_conv_output_size(h, w)
        self.fc = nn.Linear(self._to_linear, outputs)

    def _compute_conv_output_size(self, h, w):
        x = torch.rand(1, 1 if self.USE_GRAYSCALE else 3, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class LearnGame:
    def __init__(
        self,
        rom_path,
        locations,
        location_address,
        device,
        SCALE_FACTOR,
        USE_GRAYSCALE,
        goal_loc,
    ):
        self.rom_path = rom_path
        self.locations = locations
        self.location_address = location_address
        self.device = device
        self.SCALE_FACTOR = SCALE_FACTOR
        self.USE_GRAYSCALE = USE_GRAYSCALE
        self.goal_loc = goal_loc
        self.controller = Controller(self.rom_path)
        self.movements = self.controller.movements
        self.positive_keywords = {"received": False, "won": False}
        self.negative_keywords = {
            "lost": False,
            "fainted": False,
            "save the game": False,
        }

        self.screen_size = self.controller.screen_size()
        self.scaled_size = (
            int(self.screen_size[0] * SCALE_FACTOR),
            int(self.screen_size[1] * SCALE_FACTOR),
        )
        self.model = DQN(
            self.scaled_size[0],
            self.scaled_size[1],
            len(self.movements),
            self.USE_GRAYSCALE,
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayMemory(1000)
        self.checkpoint_path = "./checkpoints/pokemon_rl_checkpoint.pth"
        self.epsilon = 0.9
        self.start_episode = 0
        self.load_checkpoint()

    def save_checkpoint(self, state):
        torch.save(state, self.checkpoint_path)

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            print(f"Loading checkpoint '{self.checkpoint_path}'")
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_episode = checkpoint["epoch"]
            self.epsilon = checkpoint["epsilon"]
        else:
            print(f"No checkpoint found at '{self.checkpoint_path}'")

    def image_to_tensor(self, image):
        # Check if the image is already a PIL Image; if not, convert it
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Scale the image if needed
        if self.SCALE_FACTOR != 1:
            image = image.resize([int(s * self.SCALE_FACTOR) for s in image.size])

        # Convert to grayscale if needed
        if self.USE_GRAYSCALE:
            image = image.convert("L")

        # Convert the PIL image to a numpy array
        image = np.array(image)

        # Add an extra dimension for grayscale images
        if self.USE_GRAYSCALE:
            image = np.expand_dims(image, axis=2)

        # Convert to a PyTorch tensor, rearrange dimensions, normalize, and send to device
        image = (
            torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
            / 255
        )
        image = image.to(self.device)  # Sending tensor to the specified device

        return image

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return (
                    self.model(state).max(1)[1].view(1, 1).to(self.device)
                )  # Sending tensor to the specified device
        else:
            return torch.tensor(
                [[random.randrange(len(self.movements))]],
                dtype=torch.long,
                device=self.device,
            )  # Sending tensor to the specified device

    def optimize_model(self, batch_size=128):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = tuple(zip(*transitions))

        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])

        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(
            batch_size, device=self.device
        )  # Tensor initialized on the specified device
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch[3])),
            dtype=torch.bool,
            device=self.device,
        )  # Tensor initialized on the specified device

        non_final_next_states = torch.cat([s for s in batch[3] if s is not None])
        next_state_values[non_final_mask] = (
            self.model(non_final_next_states).max(1)[0].detach()
        )

        expected_state_action_values = (next_state_values * 0.99) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def rewards(self, action, loc, visited_locations, default_reward=0.05):
        # store if has been rewarded recently
        # if has been rewarded recently, then don't reward again

        total_reward = 0
        text = self.controller.get_text_on_screen()
        # check if any of the positive keywords are in the text
        for keyword in self.positive_keywords:
            if keyword in text:
                self.positive_keywords[keyword] = True
                total_reward += default_reward
            else:
                self.positive_keywords[keyword] = False
        # check if any of the negative keywords are in the text
        for keyword in self.negative_keywords:
            if keyword in text:
                self.negative_keywords[keyword] = True
                total_reward -= default_reward
            else:
                self.negative_keywords[keyword] = False

        # We should discourage start and select
        if action == "START" or action == "SELECT":
            total_reward -= default_reward

        # Encourage exploration
        # if loc isn't in set then reward
        if loc not in visited_locations:
            total_reward += default_reward
            # add loc to visited locations
            visited_locations.add(loc)
        else:
            # Discourage  revisiting locations too much
            total_reward -= default_reward
        return total_reward

    def run(self, num_episodes=100):
        time_per_episode = []
        for i_episode in tqdm(
            range(self.start_episode, num_episodes + self.start_episode)
        ):
            self.controller.stop()
            self.controller = Controller(self.rom_path)
            state = self.image_to_tensor(self.controller.screen_image())
            visited_locations = set()
            for t in count():
                action = self.select_action(state)
                self.controller.handleMovement(self.movements[action.item()])
                reward = torch.tensor([-0.01], dtype=torch.float32, device=self.device)

                img = self.controller.screen_image()
                loc = self.controller.get_memory_value(self.location_address)

                done = False

                reward = self.rewards(
                    self.movements[action.item()], loc, visited_locations
                )

                if self.locations[loc] == self.goal_loc:
                    reward = reward + 2
                    done = True

                next_state = self.image_to_tensor(img) if not done else None

                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                self.memory.push(state, action, reward, next_state)

                self.optimize_model()

                state = next_state
                if done:
                    time_per_episode.append(t)
                    break

            self.epsilon = max(self.epsilon * 0.99, 0.05)

            if i_episode % 5 == 0:
                self.save_checkpoint(
                    {
                        "epoch": i_episode + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epsilon": self.epsilon,
                    }
                )

        torch.save(self.model.state_dict(), "./checkpoints/pokemon_rl_model_final.pth")
        self.controller.stop(save=False)
        # Save output of list of time per episode as csv
        with open("time_per_episode.csv", "w") as f:
            write = csv.writer(f)
            write.writerow(time_per_episode)
