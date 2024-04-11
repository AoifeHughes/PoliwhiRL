# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from .episodic_memory import EpisodicMemory
from .DQN import DQNModel
from PoliwhiRL.utils import plot_best_attempts
from PoliwhiRL.environment.controller import action_space


class BaseDQNAgent:
    def __init__(self, config):
        self.config = config
        y = int(160 * config.get("scaling_factor", 1))
        x = int(144 * config.get("scaling_factor", 1))
        self.state_size = (1 if config.get("use_grayscale", False) else 3, x, y)
        self.action_size = len(action_space)
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon_start"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.memory = EpisodicMemory(config["memory_size"], config["db_path"])
        self.device = config["device"]
        self.model = DQNModel(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.criterion = nn.MSELoss()

    def epsilon_update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

    def update_model(self, batch):
        self.epsilon_update()
        max_length = max(len(episode[0]) for episode in batch)

        padded_episodes = []
        for episode in batch:
            states, actions, rewards, next_states, dones = episode
            pad_size = max_length - len(states)
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

        for i in range(len(batch)):
            for j in range(len(batch[i][0])):
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

    def plot_progress(self, rewards, id):
        plot_best_attempts(self.config["results_path"], "", id, rewards)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
