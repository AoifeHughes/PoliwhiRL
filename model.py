import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from itertools import count
import os
from tqdm import tqdm
from pyboy import PyBoy, WindowEvent
from PIL import Image
import csv

rom_path = 'Pokemon - Crystal Version.gbc'
locations = {6: "DownstairsPlayersHouse", 0: "UpstairsPlayersHouse", 4: "OutsideStartingArea"}
location_address = 0xD148

SCALE_FACTOR = 0.25
USE_GRAYSCALE = False
movements = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]

event_dict_press = {
        "UP": WindowEvent.PRESS_ARROW_UP,
        "DOWN": WindowEvent.PRESS_ARROW_DOWN,
        "LEFT": WindowEvent.PRESS_ARROW_LEFT,
        "RIGHT": WindowEvent.PRESS_ARROW_RIGHT,
        "A": WindowEvent.PRESS_BUTTON_A,
        "B": WindowEvent.PRESS_BUTTON_B,
        "START": WindowEvent.PRESS_BUTTON_START,
        "SELECT": WindowEvent.PRESS_BUTTON_SELECT
    }

event_dict_release = {
        "UP": WindowEvent.RELEASE_ARROW_UP,
        "DOWN": WindowEvent.RELEASE_ARROW_DOWN,
        "LEFT": WindowEvent.RELEASE_ARROW_LEFT,
        "RIGHT": WindowEvent.RELEASE_ARROW_RIGHT,
        "A": WindowEvent.RELEASE_BUTTON_A,
        "B": WindowEvent.RELEASE_BUTTON_B,
        "START": WindowEvent.RELEASE_BUTTON_START,
        "SELECT": WindowEvent.RELEASE_BUTTON_SELECT
    }

def start_pyboy():
    pyboy = PyBoy(rom_path, window_scale=1)
    pyboy.set_emulation_speed(target_speed=0)
    return pyboy

def pyBoyHandleMovement(pyboy, movement, ticks_per_input=30, wait=60):
    pyboy.send_input(event_dict_press[movement])
    [pyboy.tick() for _ in range(ticks_per_input)]
    pyboy.send_input(event_dict_release[movement])
    [pyboy.tick() for _ in range(wait)]


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
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
        x = torch.rand(1, 1 if USE_GRAYSCALE else 3, h, w)
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

def image_to_tensor(image):
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

    # Convert to a PyTorch tensor and rearrange dimensions
    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
    image = image.to(torch.float32) / 255  # Normalize to the range [0, 1]

    return image

pyboy = start_pyboy()
screen_size = pyboy.botsupport_manager().screen().screen_ndarray().shape[:2]
scaled_size = (int(screen_size[0] * SCALE_FACTOR), int(screen_size[1] * SCALE_FACTOR))

model = DQN(scaled_size[0], scaled_size[1], len(movements))
optimizer = optim.Adam(model.parameters(), lr=0.001)
memory = ReplayMemory(10000)

checkpoint_path = "pokemon_rl_checkpoint.pth"

def save_checkpoint(state, filename=checkpoint_path):
    torch.save(state, filename)

def load_checkpoint():
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_episode = checkpoint['epoch']
        epsilon = checkpoint['epsilon']
        return start_episode, epsilon
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, 0.9

epsilon = 0.9
start_episode, epsilon = load_checkpoint()

def select_action(state):
    global epsilon
    if random.random() > epsilon:
        with torch.no_grad():
            return model(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(len(movements))]], dtype=torch.long)

def optimize_model(batch_size=128):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = tuple(zip(*transitions))

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch[3] if s is not None])
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * 0.99) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 100
time_per_episode = []
for i_episode in tqdm(range(start_episode, num_episodes + start_episode)):
    pyboy.stop(save=False)
    pyboy = start_pyboy()
    state = image_to_tensor(pyboy.screen_image())
    visited_locations = set()
    for t in count():
        action = select_action(state)
        pyBoyHandleMovement(pyboy, movements[action.item()])
        reward = torch.tensor([-0.01], dtype=torch.float32)

        img = pyboy.screen_image()
        loc = pyboy.get_memory_value(location_address)

        done = False

        # Encourage exploration
        # if loc isn't in set then reward  
        if loc not in visited_locations:
            reward = torch.tensor([0.5], dtype=torch.float32)
            # add loc to visited locations
            visited_locations.add(loc)
        # Check for final loc = 4 
        if loc == 4:
            reward = torch.tensor([1.0], dtype=torch.float32)
            done = True

        next_state = image_to_tensor(img) if not done else None

        memory.push(state, action, reward, next_state)

        optimize_model()

        state = next_state
        if done:
            time_per_episode.append(t)
            break

    epsilon = max(epsilon * 0.99, 0.05)
   
    if i_episode % 10 == 0:
        save_checkpoint({
            'epoch': i_episode + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epsilon': epsilon
        })

torch.save(model.state_dict(), "pokemon_rl_model_final.pth")
pyboy.stop(save=False)
# Save output of list of time per episode as csv 
with open('time_per_episode.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(time_per_episode)