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
        
        # Initialize storage arrays - use episode length as buffer size, not update frequency
        buffer_size = max(self.config.get("episode_length", 800), self.update_frequency * 2)
        
        self.states = np.zeros(
            (buffer_size,) + self.input_shape, dtype=np.uint8
        )
        self.actions = np.zeros(buffer_size, dtype=np.uint8)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        
        # Adaptive exploration tensor size based on episode length
        episode_length = self.config.get("episode_length", 500)
        if episode_length > 10000:  # Long episodes
            exploration_memory_size = self.config.get("exploration_memory_size", 1000)
        else:  # Regular episodes
            exploration_memory_size = self.config.get("exploration_memory_size", 100)
            
        self.exploration_tensors = np.zeros(
            (buffer_size, exploration_memory_size, 1 + self.ppo_exploration_history_length),
            dtype=np.float32,
        )
        
        # Storage for game state variables - store as list since they're dictionaries
        self.game_states = [None] * buffer_size
        
        # Store buffer size for bounds checking
        self.buffer_size = buffer_size
        
        self.last_next_state = None
        self.episode_length = 0

    def store_transition(
        self, state, next_state, action, reward, done, log_prob, value=None, exploration_tensor=None, game_state=None
    ):
        """Store a single transition in the buffer."""
        idx = self.episode_length
        
        # Bounds check
        if idx >= self.buffer_size:
            print(f"Warning: Buffer overflow at index {idx}. Buffer size: {self.buffer_size}")
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
                    (self.buffer_size, new_locations, new_features),
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
        Get all stored data formatted for PPO training with sequence chunking.
        Returns None if not enough data for sequences.
        """
        if self.episode_length < self.sequence_length:
            return None
            
        # For long sequences, use chunking to avoid memory issues
        max_sequences_per_update = self.config.get("max_sequences_per_update", 32)
        min_sequences_per_update = self.config.get("min_sequences_per_update", 8)
        
        num_sequences = self.episode_length - self.sequence_length + 1
        
        # Limit number of sequences to prevent memory overflow
        if num_sequences > max_sequences_per_update:
            # Sample evenly across the episode
            step_size = max(1, num_sequences // max_sequences_per_update)
            sequence_indices = list(range(0, num_sequences, step_size))[:max_sequences_per_update]
        else:
            sequence_indices = list(range(num_sequences))
        
        # Ensure we have at least minimum sequences
        if len(sequence_indices) < min_sequences_per_update:
            return None
            
        # Create sequences for states - use chunking to avoid memory issues
        sequences = []
        next_sequences = []
        
        for i in sequence_indices:
            sequences.append(self.states[i : i + self.sequence_length])
            if i < num_sequences - 1:
                next_sequences.append(self.states[i + 1 : i + self.sequence_length + 1])
            else:
                next_sequences.append(
                    np.concatenate(
                        [self.states[-self.sequence_length + 1 :], [self.last_next_state]]
                    )
                )
        
        sequences = np.array(sequences)
        next_sequences = np.array(next_sequences)
        
        # Get corresponding actions, rewards, etc. for the selected sequences
        selected_actions = self.actions[self.sequence_length - 1 : self.episode_length]
        selected_rewards = self.rewards[self.sequence_length - 1 : self.episode_length]
        selected_dones = self.dones[self.sequence_length - 1 : self.episode_length]
        selected_log_probs = self.log_probs[self.sequence_length - 1 : self.episode_length]
        selected_values = self.values[self.sequence_length - 1 : self.episode_length]
        selected_exploration_tensors = self.exploration_tensors[self.sequence_length - 1 : self.episode_length]
        selected_game_states = self.game_states[self.sequence_length - 1 : self.episode_length]
        
        # For long sequences, subsample the output arrays to match sequence selection
        if len(sequence_indices) < len(selected_actions):
            # Map sequence indices to output indices
            output_indices = [idx + self.sequence_length - 1 for idx in sequence_indices]
            output_indices = [idx for idx in output_indices if idx < self.episode_length]
            
            selected_actions = selected_actions[:len(output_indices)]
            selected_rewards = selected_rewards[:len(output_indices)]
            selected_dones = selected_dones[:len(output_indices)]
            selected_log_probs = selected_log_probs[:len(output_indices)]
            selected_values = selected_values[:len(output_indices)]
            selected_exploration_tensors = selected_exploration_tensors[:len(output_indices)]
            selected_game_states = selected_game_states[:len(output_indices)]
        
        return {
            "states": torch.FloatTensor(sequences).to(self.device),
            "next_states": torch.FloatTensor(next_sequences).to(self.device),
            "actions": torch.LongTensor(selected_actions).to(self.device),
            "rewards": torch.FloatTensor(selected_rewards).to(self.device),
            "dones": torch.BoolTensor(selected_dones).to(self.device),
            "old_log_probs": torch.FloatTensor(selected_log_probs).to(self.device),
            "values": torch.FloatTensor(selected_values).to(self.device),
            "exploration_tensors": torch.FloatTensor(selected_exploration_tensors).to(self.device),
            "game_states": selected_game_states,
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