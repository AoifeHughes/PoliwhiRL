# -*- coding: utf-8 -*-

import multiprocessing as mp
from multiprocessing import Queue
import random
import numpy as np
import torch
from tqdm import tqdm

from .base_agent import BaseDQNAgent
from PoliwhiRL.environment.controller import Controller as Env


class ParallelDQNAgent(BaseDQNAgent):
    def __init__(self, config, workers_override=None):
        super().__init__(config)
        self.num_workers = (
            config["num_workers"] if workers_override is None else workers_override
        )
        self.workers = []
        self.reward_queues = [Queue() for _ in range(self.num_workers)]
        self.work_queues = [Queue() for _ in range(self.num_workers)]
        for idx in range(self.num_workers):
            worker = Worker(
                self.model,
                self.memory,
                self.config,
                idx,
                self.reward_queues[idx],
                self.work_queues[idx],
                record=(True if idx == 0 else False),
            )
            self.workers.append(worker)

    def train(self, num_episodes):
        rewards = []

        # Start the worker processes
        for worker in self.workers:
            worker.start()

        for episode in tqdm(range(num_episodes), desc="Training"):
            # Signal the workers to start a new episode
            for queue in self.work_queues:
                queue.put(("run", None))

            total_rewards = []
            for reward_queue in self.reward_queues:
                total_rewards.append(reward_queue.get())

            rewards.extend(total_rewards)
            episodes = self.memory.sample(self.batch_size)
            self.update_model(episodes)

            for worker in self.workers:
                worker.model.load_state_dict(self.model.state_dict())

            self.plot_progress(rewards, self.config["record_id"])

        # Signal the workers to terminate
        for queue in self.work_queues:
            queue.put(("terminate", None))

        # Wait for the worker processes to finish
        for worker in self.workers:
            worker.join()

    def populate_db(self, num_episodes):
        # Start the worker processes
        for worker in self.workers:
            worker.start()
        # Signal the workers to start populating the database
        for queue in self.work_queues:
            queue.put(("populate", num_episodes))

        # Wait for the worker processes to finish
        for worker in self.workers:
            worker.join()


class Worker(mp.Process):
    def __init__(
        self, model, memory, config, worker_id, reward_queue, work_queue, record=False
    ):
        super().__init__()
        self.model = model
        self.memory = memory
        self.config = config
        self.worker_id = worker_id
        self.reward_queue = reward_queue
        self.work_queue = work_queue
        self.epsilon = config["epsilon_start"]
        self.record = record

    def run(self):
        env = Env(self.config)

        while True:
            signal, data = self.work_queue.get()

            if signal == "terminate":
                break
            elif signal == "populate":
                num_episodes = data
                self.populate_db(env, num_episodes)
                break
            elif signal == "run":
                state = env.reset()
                done = False
                episode_reward = 0

                while not done:
                    action = self.act(np.array([state]))
                    next_state, reward, done = env.step(action)
                    self.memory.add(
                        state, action, reward, next_state, done, self.worker_id
                    )
                    state = next_state
                    episode_reward += reward
                    if self.record:
                        env.record(self.epsilon, f"dqn{self.worker_id}", 0, reward)
                self.reward_queue.put(episode_reward)
                self.epsilon = self.epsilon * self.config["epsilon_decay"]

        env.close()

    def populate_db(self, env, num_episodes):
        for _ in tqdm(
            range(num_episodes), desc=f"Worker {self.worker_id} - Populating DB"
        ):
            state = env.reset()
            env.extend_timeout(self.config["random_episode_length"])
            done = False
            steps = 0

            while not done:
                action = random.randrange(self.config["action_size"])
                next_state, reward, done = env.step(action)
                steps += 1
                self.memory.add(state, action, reward, next_state, done, self.worker_id)
                state = next_state

    def act(self, state_sequence, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() <= epsilon:
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
