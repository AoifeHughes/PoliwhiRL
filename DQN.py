# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import random
from multiprocessing import Manager
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self, h, w, outputs, USE_GRAYSCALE):
        super(DQN, self).__init__()
        self.USE_GRAYSCALE = USE_GRAYSCALE
        # Convolutional layers
        self.conv1 = nn.Conv2d(1 if USE_GRAYSCALE else 3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2
        )  # Additional convolutional layer
        self.bn4 = nn.BatchNorm2d(64)

        self._to_linear = None
        self._compute_conv_output_size(h, w)
        self.fc1 = nn.Linear(self._to_linear, 512)  # Larger fully connected layer
        self.fc2 = nn.Linear(512, outputs)  # Additional fully connected layer

    def _compute_conv_output_size(self, h, w):
        x = torch.rand(1, 1 if self.USE_GRAYSCALE else 3, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayMemory(object):
    def __init__(self, capacity, n_steps=5, multiCPU=True):
        manager = Manager()
        self.memory = manager.list() if multiCPU else []
        self.lock = manager.Lock() if multiCPU else DummyLock()
        self.capacity = capacity
        self.n_steps = n_steps

    def push(self, *args):
        """Saves a transition."""
        with self.lock:
            self.memory.append(args)
            if len(self.memory) > self.capacity:
                self.memory.pop(0)          
    def sample(self, batch_size):
        with self.lock:
            return [
                self.memory[i]
                for i in np.random.choice(
                    np.arange(len(self.memory)), batch_size, replace=False
                )
            ]

    def __len__(self):
        with self.lock:
            return len(self.memory)


class DummyLock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def optimize_model(
    batch_size,
    device,
    memory,
    primary_model,
    target_model,
    optimizer,
    GAMMA=0.9,
    n_steps=5,
):
    # Sample a batch of n-step sequences
    sequences = memory.sample(batch_size)

    # Initialize lists for states, actions, rewards, and next states
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    non_final_mask = []
    # each sequence is a list of [1][4]
    for sequence in sequences:
        s = sequence[0]
        cumulative_reward = sum((GAMMA**i) * np.clip(s[i][2], -1, 1) for i in range(len(s)))
        reward_batch.append(cumulative_reward)
        state_batch.append(s[0][0])
        action_batch.append(s[0][1])
        next_state = s[-1][3] if s[-1][3] is not None else None
        next_state_batch.append(next_state)
        non_final_mask.append(next_state is not None)

    # Convert lists to tensors
    state_batch = torch.cat(state_batch)
    action_batch = torch.cat(action_batch)
    reward_batch = torch.tensor(reward_batch, device=device)
    non_final_next_states = torch.cat([s for s in next_state_batch if s is not None])
    non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.bool)

    # Compute Q(s_t, a) using the primary_model
    state_action_values = primary_model(state_batch).gather(1, action_batch)

    # Initialize next state values to zero
    next_state_values = torch.zeros(batch_size, device=device)
    # Compute V(s_{t+n}) for all next states using the target_model
    if sum(non_final_mask) > 0:
        next_state_values[non_final_mask] = (
            target_model(non_final_next_states).max(1)[0].detach()
        )

    # Compute the expected Q values
    expected_state_action_values = (
        next_state_values * (GAMMA**n_steps)
    ) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in primary_model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
