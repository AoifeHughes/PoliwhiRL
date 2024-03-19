# -*- coding: utf-8 -*-
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Store priorities

    def add(self, batch, error):
        # Assume batch is a tuple of states, actions, rewards, next_states, dones
        # and error is the maximum or average error of the batch
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(batch)
        else:
            self.buffer[self.pos] = batch

        self.priorities[self.pos] = max(max_prio, error)  # Assign batch priority
        self.pos = (self.pos + 1) % self.capacity
        return self.priorities[self.pos - 1]


    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        sampled_batches = [self.buffer[idx] for idx in indices]

        # Assuming each batch is structured as (states, actions, rewards, next_states, dones)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for batch in sampled_batches:
            b_states, b_actions, b_rewards, b_next_states, b_dones = batch
            states.extend(b_states)
            actions.extend(b_actions)
            rewards.extend(b_rewards)
            next_states.extend(b_next_states)
            dones.extend(b_dones)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return states, actions, rewards, next_states, dones, indices, weights


    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        """Returns a state dictionary for checkpointing."""
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "buffer": self.buffer,
            "pos": self.pos,
            "priorities": self.priorities,
        }

    def load_state_dict(self, state_dict):
        """Loads the buffer's state from a state dictionary."""
        self.capacity = state_dict["capacity"]
        self.alpha = state_dict["alpha"]
        self.buffer = state_dict["buffer"]
        self.pos = state_dict["pos"]
        self.priorities = state_dict["priorities"]
