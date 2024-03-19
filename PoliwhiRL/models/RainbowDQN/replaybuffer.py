# -*- coding: utf-8 -*-
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Store priorities

    def add(self, sequence, error):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(sequence)
        else:
            self.buffer[self.pos] = sequence

        self.priorities[self.pos] = max(max_prio, error)  # Assign batch priority
        self.pos = (self.pos + 1) % self.capacity
        return self.priorities[self.pos - 1]

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]  # Only consider non-zero priorities

        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        # Unpack sequences
        (
            state_sequences,
            action_sequences,
            reward_sequences,
            next_state_sequences,
            done_sequences,
        ) = zip(*samples)
        return (
            state_sequences,
            action_sequences,
            reward_sequences,
            next_state_sequences,
            done_sequences,
            indices,
            weights,
        )

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error.mean()

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
