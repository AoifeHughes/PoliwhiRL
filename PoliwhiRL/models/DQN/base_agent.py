# -*- coding: utf-8 -*-
import numpy as np
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

    def update_model(self, episodes, tbptt_steps=100):
        # Group episodes by sequence length
        grouped_episodes = {}
        for episode in episodes:
            seq_length = len(episode["state"])
            if seq_length not in grouped_episodes:
                grouped_episodes[seq_length] = []
            grouped_episodes[seq_length].append(episode)

        for seq_length, episode_group in grouped_episodes.items():

            states = torch.stack([episode["state"] for episode in episode_group], dim=0).to(self.device)
            actions = torch.stack([episode["action"] for episode in episode_group], dim=0).to(self.device)
            rewards = torch.stack([episode["reward"] for episode in episode_group], dim=0).to(self.device)
            next_states = torch.stack([episode["next_state"] for episode in episode_group], dim=0).to(self.device)
            dones = torch.stack([episode["done"] for episode in episode_group], dim=0).to(self.device)

            # Truncated Backpropagation Through Time
            for start in range(0, seq_length, tbptt_steps):
                end = min(start + tbptt_steps, seq_length)

                states_tbptt = states[:, start:end]
                actions_tbptt = actions[:, start:end]
                rewards_tbptt = rewards[:, start:end]
                next_states_tbptt = next_states[:, start:end]
                dones_tbptt = dones[:, start:end]

                q_values = self.model(states_tbptt)
                next_q_values = self.model(next_states_tbptt)

                targets = q_values.clone()

                for j in range(end - start):
                    mask = dones_tbptt[:, j]
                    mask_indices = torch.where(mask)[0]
                    targets[mask_indices, j, actions_tbptt[mask_indices, j]] = rewards_tbptt[mask_indices, j]

                    not_mask = torch.logical_not(mask)
                    not_mask_indices = torch.where(not_mask)[0]
                    targets[not_mask_indices, j, actions_tbptt[not_mask_indices, j]] = (
                        rewards_tbptt[not_mask_indices, j] + self.gamma * next_q_values[not_mask_indices, j].max(dim=1)[0]
                    )

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
