# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import pickle


class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_out_size(state_size)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.relu3 = nn.ReLU()

        self.lstm = nn.LSTM(512, 128)
        self.fc2 = nn.Linear(128, action_size)

    def _get_conv_out_size(self, state_size):
        x = torch.zeros(1, *state_size)
        x = self.conv1(x)
        x = self.conv2(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        # x has shape [sequence, c, h, w]
        sequence_length = x.size(0)
        x = x.view(
            sequence_length, x.size(-3), x.size(-2), x.size(-1)
        )  # Reshape to [sequence, c, h, w]

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x, _ = self.lstm(x.unsqueeze(1))  # Add a dimension for LSTM input
        x = self.fc2(x.squeeze(1))  # Remove the extra dimension after LSTM
        return x


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
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = EpisodicMemory(memory_size)
        self.device = device
        self.model = DQNModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state_sequence):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_sequence = torch.tensor(
            np.transpose(state_sequence, (0, 3, 1, 2)), dtype=torch.float32
        ).to(self.device)
        q_values = self.model(state_sequence)[-1]
        action_probs = torch.softmax(q_values, dim=-1)
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def replay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if len(self.memory) < self.batch_size:
            return

        episodes = self.memory.sample(self.batch_size)

        for episode in episodes:
            states, actions, rewards, next_states, dones = zip(*episode)
            states = torch.tensor(
                np.transpose(np.stack(states), (0, 3, 1, 2)), dtype=torch.float32
            ).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(
                np.transpose(np.stack(next_states), (0, 3, 1, 2)), dtype=torch.float32
            ).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

            q_values = self.model(states)
            next_q_values = self.model(next_states)
            targets = q_values.clone()

            for i in range(len(episode)):
                if dones[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    targets[i, actions[i]] = rewards[i] + self.gamma * torch.max(
                        next_q_values[i]
                    )

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, targets)
            loss.backward()
            self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


class EpisodicMemory:
    def __init__(self, memory_size, db_path="./episodic_memory.db"):
        self.memory_size = memory_size
        self.db_path = db_path
        self.current_episode = []
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """CREATE TABLE IF NOT EXISTS episodes
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          states BLOB,
                          actions BLOB,
                          rewards BLOB,
                          next_states BLOB,
                          dones BLOB,
                          total_reward REAL)"""
            )
            conn.commit()

    def add(self, state, action, reward, next_state, done):
        self.current_episode.append((state, action, reward, next_state, done))
        if done:
            states, actions, rewards, next_states, dones = zip(*self.current_episode)
            states = np.stack(states)
            next_states = np.stack(next_states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            dones = np.array(dones)
            total_reward = np.sum(rewards)

            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    """INSERT INTO episodes (states, actions, rewards, next_states, dones, total_reward)
                             VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        pickle.dumps(states),
                        pickle.dumps(actions),
                        pickle.dumps(rewards),
                        pickle.dumps(next_states),
                        pickle.dumps(dones),
                        total_reward,
                    ),
                )
                conn.commit()

            self.current_episode = []

            # Limit the number of stored episodes to memory_size
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    "DELETE FROM episodes WHERE id NOT IN (SELECT id FROM episodes ORDER BY id DESC LIMIT ?)",
                    (self.memory_size,),
                )
                conn.commit()

    def sample(self, batch_size):
        half_batch_size = batch_size // 2

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()

            # Retrieve half the batch from the best-rewarded episodes
            c.execute(
                """SELECT states, actions, rewards, next_states, dones
                         FROM episodes
                         ORDER BY total_reward DESC
                         LIMIT ?""",
                (half_batch_size,),
            )
            best_rows = c.fetchall()

            # Retrieve the other half randomly
            c.execute(
                """SELECT states, actions, rewards, next_states, dones
                         FROM episodes
                         ORDER BY RANDOM()
                         LIMIT ?""",
                (batch_size - half_batch_size,),
            )
            random_rows = c.fetchall()

        episodes = []
        for row in best_rows + random_rows:
            states = pickle.loads(row[0])
            actions = pickle.loads(row[1])
            rewards = pickle.loads(row[2])
            next_states = pickle.loads(row[3])
            dones = pickle.loads(row[4])
            episode = list(zip(states, actions, rewards, next_states, dones))
            episodes.append(episode)

        return episodes

    def __len__(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM episodes")
            count = c.fetchone()[0]
        return count
