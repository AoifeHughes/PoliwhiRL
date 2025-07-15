# -*- coding: utf-8 -*-
"""
Simple in-memory PPO buffer - replaces database-based PPOMemory
Stores transitions in numpy arrays for one update cycle, then discards them.
"""
import numpy as np
import torch


class InMemoryPPOBuffer:
    """
    Enhanced in-memory buffer for PPO training data with experience replay.
    Stores states, actions, rewards, values, etc. in numpy arrays.
    Includes replay buffer to retain successful trajectories and prevent catastrophic forgetting.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.update_frequency = config["ppo_update_frequency"]
        self.sequence_length = config["sequence_length"]
        self.input_shape = config["input_shape"]
        self.ppo_exploration_history_length = config["ppo_exploration_history_length"]
        self.episode_length = 0
        
        # Experience replay buffer configuration
        self.replay_buffer_size = config.get("replay_buffer_size", 5000)
        self.replay_ratio = config.get("replay_ratio", 0.2)  # 20% of training data from replay
        self.min_replay_reward = config.get("min_replay_reward", 5.0)  # Minimum reward to store
        
        # Initialize replay buffer
        from collections import deque
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        
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
        
        # Storage for game state variables - store as list since they're dictionaries
        self.game_states = [None] * self.update_frequency
        
        self.last_next_state = None
        self.episode_length = 0

    def store_transition(
        self, state, next_state, action, reward, done, log_prob, value=None, exploration_tensor=None, game_state=None
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
            # Handle size mismatch gracefully - resize storage if needed for both dimensions
            expected_shape = self.exploration_tensors.shape[1:]  # (locations, features)
            actual_shape = exploration_tensor.shape
            
            if actual_shape != expected_shape:
                # Resize exploration tensors to match the incoming tensor
                new_locations, new_features = actual_shape
                old_tensors = self.exploration_tensors
                self.exploration_tensors = np.zeros(
                    (self.update_frequency, new_locations, new_features),
                    dtype=np.float32,
                )
                # Copy existing data up to the minimum size if we have any
                if idx > 0:
                    min_locations = min(old_tensors.shape[1], new_locations)
                    min_features = min(old_tensors.shape[2], new_features)
                    self.exploration_tensors[:idx, :min_locations, :min_features] = old_tensors[:idx, :min_locations, :min_features]
            
            self.exploration_tensors[idx] = exploration_tensor
        
        # Store game state
        if game_state is not None:
            self.game_states[idx] = game_state.copy() if isinstance(game_state, dict) else game_state
            
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
            "game_states": self.game_states[self.sequence_length - 1 : self.episode_length],
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