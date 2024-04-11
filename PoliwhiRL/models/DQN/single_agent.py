# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from .base_agent import BaseDQNAgent


class DQNAgent(BaseDQNAgent):
    def __init__(self, config):
        super().__init__(config)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if len(self.memory) < self.batch_size:
            return

        episodes = self.memory.sample(self.batch_size)
        self.update_model(episodes)

    def train(self, env, num_episodes, random_episodes, done_lim, record_id=0):
        rewards = []
        episode_lengths = []
        best_reward = float("-inf")

        # Training loop
        for episode in tqdm(range(num_episodes + random_episodes), desc="Training"):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            state_sequence = []
            action_sequence = []
            reward_sequence = []
            next_state_sequence = []
            done_sequence = []

            while not done:
                state_sequence.append(state)
                action = self.act(np.array(state_sequence))
                next_state, reward, done = env.step(action)
                if np.sum(reward_sequence) >= done_lim:
                    done = True
                env.record(self.epsilon, f"dqn{record_id}", 0, reward)
                action_sequence.append(action)
                reward_sequence.append(reward)
                next_state_sequence.append(next_state)
                done_sequence.append(done)
                state = next_state
                episode_reward += reward
                episode_length += 1

            rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if episode_reward > best_reward:
                best_reward = episode_reward

            # Add the entire episode to memory
            for i in range(len(action_sequence)):
                self.memorize(
                    state_sequence[i],
                    action_sequence[i],
                    reward_sequence[i],
                    next_state_sequence[i],
                    done_sequence[i],
                )

            if len(self.memory) >= self.batch_size and episode > random_episodes:
                self.replay()
            self.plot_progress(rewards, record_id)
