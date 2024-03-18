import numpy as np
import torch

class SequencedPrioritizedReplayBuffer:
    def __init__(self, capacity, device, alpha=0.6, sequence_length=4):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha 
        self.sequence_length = sequence_length
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, error):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio if error is None else error
        self.pos = (self.pos + 1) % self.capacity

    def _get_sequence(self, idx):
        # Adjust this method to create a sequence starting from idx
        sequence = self.buffer[max(0, idx - self.sequence_length + 1):idx + 1]
        # Fill the sequence if it's shorter than sequence_length
        while len(sequence) < self.sequence_length:
            sequence = [self.buffer[0]] + sequence
        return zip(*sequence)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < self.sequence_length:
            raise ValueError("Not enough samples in the buffer to sample a sequence")

        prios = self.priorities[:self.pos] if self.pos > 0 else self.priorities
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, replace=True, p=probs)
        samples = [self._get_sequence(idx) for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        states, actions, rewards, next_states, dones = map(lambda x: torch.tensor(list(x), dtype=torch.float32 if x[0] is states else torch.long if x[0] is actions else torch.float32 if x[0] is rewards else torch.float32 if x[0] is next_states else torch.bool).to(self.device), zip(*samples))

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error

    def __len__(self):
        return len(self.buffer)


    def state_dict(self):
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "sequence_length": self.sequence_length,
            "buffer": self.buffer,
            "pos": self.pos,
            "priorities": self.priorities,
        }

    def load_state_dict(self, state_dict):
        self.capacity = state_dict["capacity"]
        self.alpha = state_dict["alpha"]
        self.sequence_length = state_dict.get("sequence_length", 4)  # Default to 4 if not found
        self.buffer = state_dict["buffer"]
        self.pos = state_dict["pos"]
        self.priorities = state_dict["priorities"]
