# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
import torch


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Store priorities

    def add(self, state, action, reward, next_state, done, error):
        max_prio = (
            self.priorities.max() if self.buffer else 1.0
        )  # Max priority for new entry
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio if error is None else error
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4, device='cpu'):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        states, actions, rewards, next_states, dones = zip(*samples)
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



class NStepPrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, n_steps=3, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Store priorities
        self.n_step_buffer = deque(maxlen=n_steps)

    def add(self, state, action, reward, next_state, done, error):
        # Add to the n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_steps:
            return
        
        # Calculate n-step return and the final state
        R = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_steps)])
        final_state, final_action, _, _, final_done = self.n_step_buffer[-1]

        # The state to add is the oldest state in the buffer
        state_to_add, action_to_add = self.n_step_buffer[0][:2]

        self._add_to_buffer(state_to_add, action_to_add, R, final_state, final_done, error)

    def _add_to_buffer(self, state, action, reward, next_state, done, error):
        max_prio = self.priorities.max() if self.buffer else 1.0  # Max priority for new entry
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio if error is None else error
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4, device='cpu'):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        # Convert samples to tensors
        states, actions, rewards, next_states, dones = zip(*samples)

        # Assuming 's' is already a tensor. If not, your original approach was correct.
        states = torch.stack([s.clone().detach() for s in states]).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.stack([s.clone().detach() for s in next_states]).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, errors):
        # Ensure 'indices' is iterable
        if not isinstance(indices, (list, np.ndarray)):
            indices = [indices]
        
        # If 'errors' is a scalar (0-dimensional array), convert it to a 1-dimensional array with a single value
        if np.isscalar(errors) or errors.ndim == 0:
            errors = np.array([errors])
        
        # Update priorities
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
            # Note: n_step_buffer is not saved because it's transient and expected to be empty between training sessions
        }

    def load_state_dict(self, state_dict):
        """Loads the buffer's state from a state dictionary."""
        self.capacity = state_dict["capacity"]
        self.alpha = state_dict["alpha"]
        self.buffer = state_dict["buffer"]
        self.pos = state_dict["pos"]
        self.priorities = state_dict["priorities"]
        # n_step_buffer is not loaded for the same reason it's not saved
