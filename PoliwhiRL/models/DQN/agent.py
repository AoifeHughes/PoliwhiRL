# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from .episodic_memory import EpisodicMemory
from .DQN import DQNModel


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        batch_size,
        gamma,
        lr,
        epsilon,
        epsilon_decay,
        epsilon_min,
        memory_size,
        device,
        db_path,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = EpisodicMemory(memory_size, db_path)
        self.device = device
        self.model = DQNModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state_sequence):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_sequence = np.array(state_sequence)
        state_sequence = state_sequence.reshape(
            1, *state_sequence.shape
        )  # Add batch dimension
        state_sequence = torch.tensor(
            np.transpose(state_sequence, (0, 1, 4, 2, 3)), dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_sequence)
            action_probs = torch.softmax(
                q_values[0, -1], dim=-1
            )  # Get the Q-values for the last state in the sequence
            action = torch.multinomial(action_probs, num_samples=1).item()

        return action

    def replay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if len(self.memory) < self.batch_size:
            return

        episodes = self.memory.sample(self.batch_size)

        # Find the maximum length of the episodes
        max_length = max(len(episode) for episode in episodes)

        # Pad the episodes to the maximum length
        padded_episodes = []
        for episode in episodes:
            states, actions, rewards, next_states, dones = zip(*episode)
            pad_size = max_length - len(episode)
            states = list(states) + [np.zeros_like(states[0])] * pad_size
            actions = list(actions) + [0] * pad_size
            rewards = list(rewards) + [0] * pad_size
            next_states = list(next_states) + [np.zeros_like(next_states[0])] * pad_size
            dones = list(dones) + [True] * pad_size
            padded_episodes.append((states, actions, rewards, next_states, dones))

        states, actions, rewards, next_states, dones = zip(*padded_episodes)

        states = torch.tensor(
            np.transpose(np.stack(states), (0, 1, 4, 2, 3)), dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(np.stack(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.stack(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(
            np.transpose(np.stack(next_states), (0, 1, 4, 2, 3)), dtype=torch.float32
        ).to(self.device)
        dones = torch.tensor(np.stack(dones), dtype=torch.float32).to(self.device)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        targets = q_values.clone()

        for i in range(len(episodes)):
            for j in range(len(episodes[i])):
                if dones[i, j]:
                    targets[i, j, actions[i, j]] = rewards[i, j]
                else:
                    targets[i, j, actions[i, j]] = rewards[
                        i, j
                    ] + self.gamma * torch.max(next_q_values[i, j])

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, targets)
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
