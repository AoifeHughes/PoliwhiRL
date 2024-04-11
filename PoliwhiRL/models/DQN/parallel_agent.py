# -*- coding: utf-8 -*-
import multiprocessing as mp
import random
import numpy as np
import torch
from tqdm import tqdm
from .DQN import DQNModel
from PoliwhiRL.environment.controller import Controller as Env
from PoliwhiRL.environment.controller import action_space
import torch.nn as nn
import torch.optim as optim
from .episodic_memory import EpisodicMemory


class ParallelDQNAgent:
    def __init__(self, num_workers, state_size, action_size, batch_size, gamma, lr, epsilon, epsilon_decay,
                 epsilon_min, memory_size, device, db_path, config):
        self.num_workers = num_workers
        self.model = DQNModel(state_size, action_size).to(device)
        self.memory = EpisodicMemory(memory_size, db_path, parallel=True)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.workers = []

        for idx in range(num_workers):
            worker = Worker(self.model, self.memory, state_size, action_size, epsilon, epsilon_decay, epsilon_min, config, idx)
            self.workers.append(worker)

    def train(self, num_episodes):
        for worker in self.workers:
            worker.start()
        for episode in tqdm(range(num_episodes), desc="Training"):
            for worker in self.workers:
                worker.join()
            batch = self.memory.sample(self.batch_size)
            self.update_model(batch)
            for worker in self.workers:
                worker.model.load_state_dict(self.model.state_dict())
            for worker in self.workers:
                worker.reset()
        for worker in self.workers:
            worker.terminate()

    def update_model(self, batch):
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
                    targets[i, j, actions[i, j]] = rewards[i, j] + self.gamma * torch.max(next_q_values[i, j])

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, targets)
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

class Worker(mp.Process):
    def __init__(self, model, memory, state_size, action_size, epsilon, epsilon_decay, epsilon_min, config, worker_id):
        super().__init__()
        self.model = model
        self.memory = memory
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.worker_id = worker_id
        self.config = config

    def run(self):
        env = Env(self.config)
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action = self.act(np.array([state]))
            next_state, reward, done = env.step(action)
            self.memory.add(state, action, reward, next_state, done, self.worker_id)
            state = next_state
            episode_reward += reward
            episode_length += 1

        env.close()

    def act(self, state_sequence):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_sequence = state_sequence.reshape(1, *state_sequence.shape)  # Add batch dimension
        state_sequence = torch.tensor(np.transpose(state_sequence, (0, 1, 4, 2, 3)), dtype=torch.float32)

        with torch.no_grad():
            q_values = self.model(state_sequence)
            action = torch.argmax(q_values[0, -1]).item()

        return action
    
    def reset(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
