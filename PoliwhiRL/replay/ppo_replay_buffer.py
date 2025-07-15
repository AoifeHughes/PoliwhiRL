# -*- coding: utf-8 -*-
"""
Enhanced PPO buffer with experience replay to prevent catastrophic forgetting.
Extends the basic InMemoryPPOBuffer with replay capabilities.
"""
import numpy as np
import torch
from collections import deque
import random

from .ppo_memory_buffer import InMemoryPPOBuffer


class PPOReplayBuffer(InMemoryPPOBuffer):
    """
    Enhanced PPO buffer with experience replay capabilities.
    Stores successful trajectories and mixes them with current experience.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Experience replay configuration
        self.replay_buffer_size = config.get("replay_buffer_size", 5000)
        self.replay_ratio = config.get("replay_ratio", 0.2)  # 20% of training data from replay
        self.min_replay_reward = config.get("min_replay_reward", 5.0)  # Minimum reward to store
        self.replay_priority_alpha = config.get("replay_priority_alpha", 0.6)  # Priority weighting
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.replay_priorities = deque(maxlen=self.replay_buffer_size)
        
        # Episode tracking for replay storage
        self.current_episode_reward = 0.0
        self.current_episode_data = []
        
    def store_experience(self, state, action, reward, next_state, done, 
                        old_log_prob, value, exploration_tensor=None, game_state=None):
        """
        Store experience in both current buffer and track for replay buffer.
        """
        # Store in current buffer
        super().store_experience(state, action, reward, next_state, done, 
                               old_log_prob, value, exploration_tensor, game_state)
        
        # Track current episode data for potential replay storage
        self.current_episode_reward += reward
        self.current_episode_data.append({
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'old_log_prob': old_log_prob,
            'value': value,
            'exploration_tensor': exploration_tensor.copy() if exploration_tensor is not None else None,
            'game_state': game_state.copy() if game_state is not None else None
        })
        
        # If episode is done, consider storing in replay buffer
        if done:
            self._maybe_store_episode_in_replay()
    
    def _maybe_store_episode_in_replay(self):
        """
        Store current episode in replay buffer if it meets criteria.
        """
        if self.current_episode_reward >= self.min_replay_reward:
            # Calculate priority based on reward
            priority = max(1.0, self.current_episode_reward)
            
            # Store episode data in replay buffer
            episode_data = {
                'data': self.current_episode_data.copy(),
                'reward': self.current_episode_reward,
                'priority': priority
            }
            
            self.replay_buffer.append(episode_data)
            self.replay_priorities.append(priority)
            
            print(f"🔄 Stored episode in replay buffer (reward: {self.current_episode_reward:.2f}, "
                  f"buffer size: {len(self.replay_buffer)})")
        
        # Reset episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_data = []
    
    def get_all_data(self):
        """
        Get training data combining current episode and replay buffer.
        """
        current_data = super().get_all_data()
        if current_data is None:
            return None
        
        # Add replay data if available
        if len(self.replay_buffer) > 0 and self.replay_ratio > 0:
            replay_data = self._sample_replay_data()
            if replay_data is not None:
                current_data = self._combine_data(current_data, replay_data)
        
        return current_data
    
    def _sample_replay_data(self):
        """
        Sample data from replay buffer using priority-based sampling.
        """
        if len(self.replay_buffer) == 0:
            return None
        
        # Calculate number of replay samples
        current_batch_size = len(self.actions[:self.episode_length])
        num_replay_samples = int(current_batch_size * self.replay_ratio)
        
        if num_replay_samples == 0:
            return None
        
        # Priority-based sampling
        priorities = np.array(self.replay_priorities)
        probabilities = priorities ** self.replay_priority_alpha
        probabilities = probabilities / probabilities.sum()
        
        # Sample episodes
        sampled_indices = np.random.choice(
            len(self.replay_buffer), 
            size=min(num_replay_samples, len(self.replay_buffer)), 
            p=probabilities,
            replace=False
        )
        
        # Collect sampled data
        replay_states = []
        replay_actions = []
        replay_rewards = []
        replay_next_states = []
        replay_dones = []
        replay_log_probs = []
        replay_values = []
        replay_exploration_tensors = []
        replay_game_states = []
        
        for idx in sampled_indices:
            episode = self.replay_buffer[idx]
            # Sample random segments from the episode
            episode_data = episode['data']
            segment_length = min(len(episode_data), self.sequence_length)
            
            for i in range(0, len(episode_data), segment_length):
                segment = episode_data[i:i+segment_length]
                for step in segment:
                    replay_states.append(step['state'])
                    replay_actions.append(step['action'])
                    replay_rewards.append(step['reward'])
                    replay_next_states.append(step['next_state'])
                    replay_dones.append(step['done'])
                    replay_log_probs.append(step['old_log_prob'])
                    replay_values.append(step['value'])
                    replay_exploration_tensors.append(step['exploration_tensor'])
                    replay_game_states.append(step['game_state'])
        
        if len(replay_states) == 0:
            return None
        
        # Format replay data
        replay_data = {
            'states': torch.FloatTensor(replay_states).to(self.device),
            'actions': torch.LongTensor(replay_actions).to(self.device),
            'rewards': torch.FloatTensor(replay_rewards).to(self.device),
            'next_states': torch.FloatTensor(replay_next_states).to(self.device),
            'dones': torch.BoolTensor(replay_dones).to(self.device),
            'old_log_probs': torch.FloatTensor(replay_log_probs).to(self.device),
            'values': torch.FloatTensor(replay_values).to(self.device)
        }
        
        # Add optional data if available
        if any(et is not None for et in replay_exploration_tensors):
            replay_data['exploration_tensors'] = torch.FloatTensor([
                et if et is not None else np.zeros(self.exploration_tensor_size) 
                for et in replay_exploration_tensors
            ]).to(self.device)
        
        if any(gs is not None for gs in replay_game_states):
            replay_data['game_states'] = [
                gs if gs is not None else {} 
                for gs in replay_game_states
            ]
        
        return replay_data
    
    def _combine_data(self, current_data, replay_data):
        """
        Combine current episode data with replay buffer data.
        """
        combined_data = {}
        
        for key in current_data.keys():
            if key in replay_data:
                if isinstance(current_data[key], torch.Tensor):
                    combined_data[key] = torch.cat([current_data[key], replay_data[key]], dim=0)
                elif isinstance(current_data[key], list):
                    combined_data[key] = current_data[key] + replay_data[key]
                else:
                    combined_data[key] = current_data[key]  # Keep current if can't combine
            else:
                combined_data[key] = current_data[key]
        
        return combined_data
    
    def reset(self, config=None):
        """
        Reset current episode buffer but preserve replay buffer.
        """
        super().reset(config)
        
        # Reset episode tracking but keep replay buffer
        self.current_episode_reward = 0.0
        self.current_episode_data = []
        
        print(f"📊 Reset episode buffer. Replay buffer size: {len(self.replay_buffer)}")
    
    def get_replay_stats(self):
        """
        Get statistics about the replay buffer.
        """
        if len(self.replay_buffer) == 0:
            return {"size": 0, "avg_reward": 0.0, "max_reward": 0.0}
        
        rewards = [episode['reward'] for episode in self.replay_buffer]
        return {
            "size": len(self.replay_buffer),
            "avg_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards)
        }