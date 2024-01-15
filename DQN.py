# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from itertools import count
import os
from PIL import Image
from controls import Controller
import multiprocessing
from multiprocessing import Pool
from itertools import count
from collections import namedtuple

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


DEBUG = False


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


def run_model(
    rom_path,
    locations,
    location_address,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    goal_locs,
    goal_targets,
    timeout,
    num_episodes=20,
    report_interval=10,
    num_workers=None,
):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    controller = Controller(rom_path)
    movements = controller.movements
    positive_keywords = {"received": False, "won": False}
    negative_keywords = {
        "lost": False,
        "fainted": False,
        "save the game": False,
        "saving": False,
        "select": False,
        "town map": False,
        "text speed": False,
    }
    screen_size = controller.screen_size()
    scaled_size = (
        int(screen_size[0] * SCALE_FACTOR),
        int(screen_size[1] * SCALE_FACTOR),
    )
    shared_model = DQN(
        scaled_size[0], scaled_size[1], len(movements), USE_GRAYSCALE
    ).to(device)
    shared_model.share_memory()  # Prepare model for sharing across processes
    shared_memory = ReplayMemory(1000)
    shared_optimizer = optim.Adam(shared_model.parameters(), lr=0.005)
    default_reward = 0.1
    phase = 0
    # Load checkpoint if available
    checkpoint_path = "./checkpoints/pokemon_rl_checkpoint.pth"
    epsilon_initial = 0.9
    epsilon_decay = 0.99
    epsilon_min = 0.1
    start_episode, epsilon_initial = load_checkpoint(
        checkpoint_path, shared_model, shared_optimizer, epsilon_initial
    )
    gamma = 0.5
    batch_size = 128
    epsilon = epsilon_initial
    min_update_steps = 5
    results = []
    for start in range(0, num_episodes, report_interval):
        end = min(start + report_interval, num_episodes)
        episodes_this_round = end - start
        episodes_per_worker = episodes_this_round // num_workers

        args = [
            (
                worker_id,
                episodes_per_worker,
                rom_path,
                locations,
                location_address,
                device,
                SCALE_FACTOR,
                USE_GRAYSCALE,
                goal_locs,
                timeout,
                shared_model.state_dict(),
                epsilon,
                negative_keywords,
                positive_keywords,
                default_reward,
                phase,
                movements
            )
            for worker_id in range(num_workers)
        ]

        with Pool(num_workers) as pool:
            partial_results, run_times = pool.starmap(run_episode, args)
            new_experiences = sum(len(worker_result) for worker_result in partial_results)
        results.extend(partial_results)
        update_steps = max(min_update_steps, new_experiences // batch_size)

        # Aggregate and report results after every 'report_interval' episodes
        aggregate_results(results, shared_memory, shared_optimizer, shared_model, device, gamma, batch_size, update_steps)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Save checkpoint after every 'report_interval' episodes and record number of episodes completed
        save_checkpoint(
            {
                "epoch": start + report_interval,
                "state_dict": shared_model.state_dict(),
                "optimizer": shared_optimizer.state_dict(),
                "epsilon": epsilon,
            },
            filename=f"./checkpoints/pokemon_rl_checkpoint_{start + report_interval}.pth",
        )
        # Print the average run time per worker 
        print(f"Average run time per worker: {np.mean(run_times)}")

    return results

def save_checkpoint(state, filename="pokemon_rl_checkpoint.pth"):
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer, epsilon):
    start_episode = 0
    # Check for latest checkpoint in checkpoints folder
    tmp_path = checkpoint_path
    if os.path.isdir("./checkpoints"):
        checkpoints = [f for f in os.listdir("./checkpoints") if f.endswith(".pth")]
        if len(checkpoints) > 0:
            # sort checkpoints by last modified date
            checkpoints.sort(key=lambda x: os.path.getmtime("./checkpoints/" + x))
            checkpoint_path = "./checkpoints/" + checkpoints[-1]
            # Set the start episode to the number of the checkpoint
            start_episode = int(checkpoint_path.split("_")[-1][:-4])
    else:
        os.mkdir("./checkpoints")
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_episode = checkpoint["epoch"]
            epsilon = checkpoint["epsilon"]
        except:
            print("Failed to load checkpoint")
        checkpoint_path = tmp_path
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")

    return start_episode, epsilon


def image_to_tensor(image, SCALE_FACTOR, USE_GRAYSCALE, device):
    # Check if the image is already a PIL Image; if not, convert it
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Scale the image if needed
    if SCALE_FACTOR != 1:
        image = image.resize([int(s * SCALE_FACTOR) for s in image.size])

    # Convert to grayscale if needed
    if USE_GRAYSCALE:
        image = image.convert("L")

    # Convert the PIL image to a numpy array
    image = np.array(image)

    # Add an extra dimension for grayscale images
    if USE_GRAYSCALE:
        image = np.expand_dims(image, axis=2)

    # Convert to a PyTorch tensor, rearrange dimensions, normalize, and send to device
    image = (
        torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32) / 255
    )
    image = image.to(device)  # Sending tensor to the specified device

    return image


def select_action(state, epsilon, movements, model, device):
    if random.random() > epsilon:
        with torch.no_grad():
            return (
                model(state).max(1)[1].view(1, 1).to(device)
            )  # Sending tensor to the specified device
    else:
        return torch.tensor(
            [[random.randrange(len(movements))]],
            dtype=torch.long,
            device=device,
        )  # Sending tensor to the specified device


def optimize_model(batch, model, optimizer, device, gamma):
    # Unpack the batch of experiences
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

    non_final_mask = ~torch.tensor(done_batch, dtype=torch.bool, device=device)
    non_final_next_states = torch.cat(
        [s for s, done in zip(next_state_batch, done_batch) if not done]
    )

    state_batch = torch.cat(state_batch)
    action_batch = torch.cat(action_batch)
    reward_batch = torch.cat(reward_batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch.unsqueeze(-1))

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(len(state_batch), device=device)
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def rewards(
    action,
    loc,
    visited_locations,
    controller,
    positive_keywords,
    negative_keywords,
    default_reward,
):
    # store if has been rewarded recently
    # if has been rewarded recently, then don't reward again

    total_reward = 0
    text = controller.get_text_on_screen()
    # check if any of the positive keywords are in the text
    for keyword in positive_keywords:
        if keyword in text.lower() and not positive_keywords[keyword]:
            positive_keywords[keyword] = True
            total_reward += default_reward
            if DEBUG:
                print("Found positive keyword: ", keyword)

        else:
            positive_keywords[keyword] = False
    # check if any of the negative keywords are in the text
    for keyword in negative_keywords:
        if keyword in text.lower() and not negative_keywords[keyword]:
            negative_keywords[keyword] = True
            total_reward -= default_reward
            if DEBUG:
                print("Found negative keyword: ", keyword)
        else:
            negative_keywords[keyword] = False

    # We should discourage start and select
    if action == "START" or action == "SELECT":
        total_reward -= default_reward * 2
        if DEBUG:
            print("Discouraging start and select")
    # Encourage exploration
    # if loc isn't in set then reward
    if loc not in visited_locations:
        total_reward += default_reward
        # add loc to visited locations
        visited_locations.add(loc)
        if DEBUG:
            print("Encouraging exploration")
            print("Visited locations: ", visited_locations)
    else:
        # Discourage  revisiting locations too much
        if DEBUG:
            print("Discouraging revisiting locations")
            total_reward -= default_reward

    if DEBUG:
        print("Total reward: ", total_reward)
    return total_reward


def aggregate_results(
    results, memory, optimizer, model, device, gamma, batch_size, update_steps
):
    # Unpack results and update shared replay memory
    for worker_experiences in results:
        for experience in worker_experiences:
            memory.push(*experience)

    # Update the model based on the aggregated experiences
    if len(memory) > batch_size:
        for _ in range(update_steps):
            transitions = memory.sample(batch_size)
            batch = _transition_to_batch(transitions)
            optimize_model(batch, model, optimizer, device, gamma)


def _transition_to_batch(transitions):
    # Transforms a batch of transitions to separate batches of states, actions, etc.
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    return state_batch, action_batch, reward_batch, non_final_next_states


def run_episode(
    worker_id,
    num_episodes,
    rom_path,
    locations,
    location_address,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    goal_locs,
    timeout,
    shared_model_state,
    epsilon,
    negative_keywords,
    positive_keywords,
    default_reward,
    phase,
    movements
):
    controller = Controller(rom_path)
    screen_size = controller.screen_size()
    scaled_size = (
        int(screen_size[0] * SCALE_FACTOR),
        int(screen_size[1] * SCALE_FACTOR),
    )
    model = DQN(
        scaled_size[0], scaled_size[1], len(controller.movements), USE_GRAYSCALE
    ).to(device)
    model.load_state_dict(shared_model_state)  # Load shared model state

    time_per_episode = []
    experiences = []

    for i_episode in range(num_episodes):
        controller.stop()
        controller = Controller(rom_path)
        state = image_to_tensor(
            controller.screen_image(), SCALE_FACTOR, USE_GRAYSCALE, device
        )
        visited_locations = set()
        total_reward = 0

        for t in count():
            action = select_action(state, epsilon, movements, model, device)
            controller.handleMovement(controller.movements[action.item()])
            reward = torch.tensor([-0.01], dtype=torch.float32, device=device)
            img = controller.screen_image()
            loc = controller.get_memory_value(location_address)
            done = False

            reward_value = rewards(
                action,
                loc,
                visited_locations,
                controller,
                positive_keywords,
                negative_keywords,
                default_reward,
            )
            reward += reward_value
            if loc in locations and locations[loc] == goal_locs[phase]:
                reward += 1
                print("done")
                done = True

            next_state = (
                image_to_tensor(img, SCALE_FACTOR, USE_GRAYSCALE, device)
                if not done
                else None
            )
            experiences.append(Transition(state, action, reward, next_state))

            total_reward += reward.item()
            state = next_state
            if done:
                break

        time_per_episode.append(t)

    return experiences, time_per_episode
