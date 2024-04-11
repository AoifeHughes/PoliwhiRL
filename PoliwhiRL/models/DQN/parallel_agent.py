# -*- coding: utf-8 -*-
import multiprocessing as mp
import random
import numpy as np
import torch
from tqdm import tqdm
from .base_agent import BaseDQNAgent
from PoliwhiRL.environment.controller import Controller as Env
from .episodic_memory import EpisodicMemory


class ParallelDQNAgent(BaseDQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.num_workers = config["num_workers"]
        self.memory = EpisodicMemory(
            self.config["memory_size"], self.config["db_path"], parallel=True
        )
        self.workers = []

        for idx in range(self.num_workers):
            worker = Worker(self.model, self.memory, self.config, idx)
            self.workers.append(worker)

    def train(self, num_episodes):
        rewards = []

        for worker in self.workers:
            worker.start()

        for episode in tqdm(range(num_episodes), desc="Training"):
            total_rewards = []
            for worker in self.workers:
                worker.join()
                total_rewards.append(worker.total_reward)
                worker.reset()
            rewards.extend(total_rewards)

            batch = self.memory.sample(self.batch_size)
            self.update_model(batch)

            for worker in self.workers:
                worker.model.load_state_dict(self.model.state_dict())

        for worker in self.workers:
            worker.terminate()

        self.plot_progress(rewards, self.config["record_id"])


class Worker(mp.Process):
    def __init__(self, model, memory, config, worker_id):
        super().__init__()
        self.model = model
        self.memory = memory
        self.config = config
        self.worker_id = worker_id
        self.total_reward = 0

    def run(self):
        env = Env(self.config)
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = self.act(np.array([state]))
            next_state, reward, done = env.step(action)
            self.memory.add(state, action, reward, next_state, done, self.worker_id)
            state = next_state
            episode_reward += reward

        env.close()
        self.total_reward = episode_reward

    def act(self, state_sequence):
        if np.random.rand() <= self.config["epsilon"]:
            return random.randrange(self.config["action_size"])

        state_sequence = state_sequence.reshape(
            1, *state_sequence.shape
        )  # Add batch dimension
        state_sequence = torch.tensor(
            np.transpose(state_sequence, (0, 1, 4, 2, 3)), dtype=torch.float32
        )

        with torch.no_grad():
            q_values = self.model(state_sequence)
            action = torch.argmax(q_values[0, -1]).item()

        return action

    def reset(self):
        self.total_reward = 0
        self.config["epsilon"] = max(
            self.config["epsilon_min"],
            self.config["epsilon"] * self.config["epsilon_decay"],
        )
