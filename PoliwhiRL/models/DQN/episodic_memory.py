# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import torch
import multiprocessing as mp


class EpisodicMemory:
    def __init__(
        self, memory_size, file_path="./database/episodic_memory.h5", clear_interval=100
    ):
        self.memory_size = memory_size
        self.file_path = file_path
        self.partial_episodes = {}
        self.lock = mp.Lock()
        self.clear_interval = clear_interval
        self.add_count = 0

        print("Current memory size: ", self.memory_size)
        print("Current best reward: ", self.get_highest_reward())

    def add(self, state, action, reward, next_state, done, worker_id=0):
        if worker_id not in self.partial_episodes:
            self.partial_episodes[worker_id] = []

        self.partial_episodes[worker_id].append((state, action, reward, done))

        if done:
            self._store_episode(worker_id)
            self.partial_episodes[worker_id] = []
            self.add_count += 1
            if self.add_count % self.clear_interval == 0:
                self._clear_file()

    def _store_episode(self, worker_id):
        episode = self.partial_episodes[worker_id]
        if not episode:
            return

        states, actions, rewards, dones = zip(*episode)

        episode_data = {
            "state": torch.stack(
                [torch.from_numpy(s).permute(2, 0, 1).float() for s in states]
            ),
            "action": torch.tensor(actions),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.float32),
            "total_reward": torch.tensor(sum(rewards), dtype=torch.float32),
        }

        with self.lock:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            if not os.path.exists(self.file_path):
                with h5py.File(self.file_path, "w") as f:
                    f.attrs["num_episodes"] = 0

            with h5py.File(self.file_path, "a") as f:
                episode_id = f.attrs["num_episodes"]
                f.attrs["num_episodes"] += 1

                episode_group = f.create_group(str(episode_id))
                for key, value in episode_data.items():
                    episode_group.create_dataset(key, data=value.numpy())

            # Limit the number of stored episodes to memory_size
            with h5py.File(self.file_path, "a") as f:
                if len(f.keys()) > self.memory_size:
                    oldest_episode_id = min(int(k) for k in f.keys())
                    del f[str(oldest_episode_id)]

    def sample(self, batch_size):
        with h5py.File(self.file_path, "r") as f:
            episode_ids = list(f.keys())
            total_rewards = [
                f[episode_id]["total_reward"][()] for episode_id in episode_ids
            ]

            # Convert total rewards to probabilities using softmax
            total_rewards = np.array(total_rewards)
            probabilities = np.exp(total_rewards) / np.sum(np.exp(total_rewards))

            # Sample episode IDs based on probabilities
            sampled_episode_ids = np.random.choice(
                episode_ids, size=batch_size, replace=True, p=probabilities
            )

            episodes = []
            for episode_id in sampled_episode_ids:
                episode_group = f[episode_id]
                episode_data = {}
                for key, value in episode_group.items():
                    data = value[()]
                    if isinstance(data, np.ndarray):
                        episode_data[key] = torch.from_numpy(data)
                    else:
                        episode_data[key] = torch.tensor(data)
                episodes.append(episode_data)

        for episode_data in episodes:
            states = episode_data["state"]
            next_states = states[1:]  # Shift the state sequence by one step

            # Handle the last step separately
            last_state = states[-1]
            last_next_state = torch.zeros_like(
                last_state
            )  # Placeholder for the last next_state

            next_states = torch.cat((next_states, last_next_state.unsqueeze(0)))

            episode_data["next_state"] = next_states

        return episodes

    def get_highest_reward(self):
        highest_reward = float("-inf")

        with h5py.File(self.file_path, "r") as f:
            for episode_id in f.keys():
                episode_reward = f[episode_id]["total_reward"][()]
                highest_reward = max(highest_reward, episode_reward)

        return highest_reward

    def _clear_file(self):
        with self.lock:
            temp_file_path = self.file_path + ".temp"
            with h5py.File(temp_file_path, "w") as temp_f:
                with h5py.File(self.file_path, "r") as f:
                    episode_ids = list(f.keys())
                    num_episodes = len(episode_ids)
                    start_index = max(0, num_episodes - self.memory_size)

                    for episode_id in episode_ids[start_index:]:
                        temp_f.copy(f[episode_id], episode_id)

                    temp_f.attrs["num_episodes"] = min(num_episodes, self.memory_size)

            os.replace(temp_file_path, self.file_path)

    def __len__(self):
        with h5py.File(self.file_path, "r") as f:
            if "num_episodes" in f.attrs:
                return f.attrs["num_episodes"]
            else:
                return 0
