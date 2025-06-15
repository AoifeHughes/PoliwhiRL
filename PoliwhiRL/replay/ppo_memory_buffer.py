# -*- coding: utf-8 -*-
"""
Simple in-memory PPO buffer - replaces database-based PPOMemory
Stores transitions in numpy arrays for one update cycle, then discards them.
"""
import numpy as np
import torch


class InMemoryPPOBuffer:
    """
    Simple in-memory buffer for PPO training data.
    Stores states, actions, rewards, values, etc. in numpy arrays.
    No persistence - data is discarded after each update cycle.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.update_frequency = config["ppo_update_frequency"]
        self.sequence_length = config["sequence_length"]
        self.input_shape = config["input_shape"]
        self.ppo_exploration_history_length = config["ppo_exploration_history_length"]
        self.episode_length = 0
        self.reset()

    def reset(self, config=None):
        """Reset the buffer, clearing all stored data."""
        if config is not None:
            # Update config if provided
            self.config = config
            self.device = torch.device(config["device"])
            self.update_frequency = config["ppo_update_frequency"]
            self.sequence_length = config["sequence_length"]
            self.input_shape = config["input_shape"]
            self.ppo_exploration_history_length = config["ppo_exploration_history_length"]
        
        # Initialize storage arrays
        self.states = np.zeros(
            (self.update_frequency,) + self.input_shape, dtype=np.uint8
        )
        self.actions = np.zeros(self.update_frequency, dtype=np.uint8)
        self.rewards = np.zeros(self.update_frequency, dtype=np.float32)
        self.dones = np.zeros(self.update_frequency, dtype=np.bool_)
        self.log_probs = np.zeros(self.update_frequency, dtype=np.float32)
        self.values = np.zeros(self.update_frequency, dtype=np.float32)
        
        # Adaptive exploration tensor size based on episode length
        episode_length = self.config.get("episode_length", 500)
        if episode_length > 10000:  # Long episodes
            exploration_memory_size = self.config.get("exploration_memory_size", 1000)
        else:  # Regular episodes
            exploration_memory_size = self.config.get("exploration_memory_size", 100)
            
        self.exploration_tensors = np.zeros(
            (self.update_frequency, exploration_memory_size, 1 + self.ppo_exploration_history_length),
            dtype=np.float32,
        )
        self.last_next_state = None
        self.episode_length = 0

    def store_transition(
        self, state, next_state, action, reward, done, log_prob, value=None, exploration_tensor=None
    ):
        """Store a single transition in the buffer."""
        idx = self.episode_length
        
        # Bounds check
        if idx >= self.update_frequency:
            print(f"Warning: Buffer overflow at index {idx}. Buffer size: {self.update_frequency}")
            return
            
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        
        if value is not None:
            self.values[idx] = value
            
        if exploration_tensor is not None:
            # Handle size mismatch gracefully - resize storage if needed
            if exploration_tensor.shape[0] != self.exploration_tensors.shape[1]:
                # Resize exploration tensors to match the incoming tensor
                new_size = exploration_tensor.shape[0]
                old_tensors = self.exploration_tensors
                self.exploration_tensors = np.zeros(
                    (self.update_frequency, new_size, 1 + self.ppo_exploration_history_length),
                    dtype=np.float32,
                )
                # Copy existing data up to the minimum size
                min_size = min(old_tensors.shape[1], new_size)
                self.exploration_tensors[:idx, :min_size, :] = old_tensors[:idx, :min_size, :]
            
            self.exploration_tensors[idx] = exploration_tensor
            
        self.last_next_state = next_state
        self.episode_length += 1

    def get_all_data(self):
        """
        Get all stored data formatted for PPO training.
        Returns None if not enough data for sequences.
        """
        if self.episode_length < self.sequence_length:
            return None
            
        num_sequences = self.episode_length - self.sequence_length + 1
        
        # Create sequences for states
        sequences = np.array(
            [self.states[i : i + self.sequence_length] for i in range(num_sequences)]
        )
        
        # Create next state sequences
        next_sequences = np.array(
            [
                self.states[i + 1 : i + self.sequence_length + 1]
                for i in range(num_sequences - 1)
            ]
            + [
                np.concatenate(
                    [self.states[-self.sequence_length + 1 :], [self.last_next_state]]
                )
            ]
        )
        
        return {
            "states": torch.FloatTensor(sequences).to(self.device),
            "next_states": torch.FloatTensor(next_sequences).to(self.device),
            "actions": torch.LongTensor(
                self.actions[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "rewards": torch.FloatTensor(
                self.rewards[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "dones": torch.BoolTensor(
                self.dones[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "old_log_probs": torch.FloatTensor(
                self.log_probs[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "values": torch.FloatTensor(
                self.values[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "exploration_tensors": torch.FloatTensor(
                self.exploration_tensors[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
        }

    def __len__(self):
        """Return the number of stored transitions."""
        return self.episode_length
    
    # Compatibility methods for any code that might expect database functionality
    @staticmethod
    def get_memory_ids(config):
        """Return empty list - no database to query."""
        return []
    
    @staticmethod
    def load_from_database(config, memory_id):
        """Return None - no database to load from."""
        return None