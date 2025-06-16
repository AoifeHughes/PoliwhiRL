# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque, Counter
import pickle


class MacroActionModule(nn.Module):
    """
    Learns and executes macro actions (sequences of primitive actions).
    
    Automatically discovers repetitive action patterns and learns to execute
    them as single macro actions for more efficient navigation and interaction.
    """
    
    def __init__(self, action_space_size, max_sequence_length=5, d_model=256, device='cpu'):
        """
        Initialize macro action learning module.
        
        Args:
            action_space_size: Number of primitive actions
            max_sequence_length: Maximum length of macro action sequences
            d_model: Model dimension for sequence embedding
            device: Device for computations
        """
        super(MacroActionModule, self).__init__()
        
        self.action_space_size = action_space_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.device = torch.device(device)
        
        # Macro action vocabulary
        self.macro_vocab = {}  # sequence -> macro_id mapping
        self.macro_sequences = {}  # macro_id -> sequence mapping
        self.macro_success_rates = defaultdict(float)  # macro_id -> success rate
        self.macro_usage_counts = defaultdict(int)  # macro_id -> usage count
        self.next_macro_id = action_space_size  # Start macro IDs after primitive actions
        
        # Pattern detection
        self.recent_actions = deque(maxlen=max_sequence_length * 3)
        self.pattern_candidates = defaultdict(int)  # pattern -> frequency
        self.min_pattern_frequency = 3  # Minimum occurrences to become macro
        
        # Sequence embedding network
        self.sequence_encoder = nn.Sequential(
            nn.Embedding(action_space_size + 1000, 64),  # +1000 for potential macro actions
            nn.LSTM(64, 128, batch_first=True),
        )
        
        # Action sequence predictor
        self.sequence_predictor = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, action_space_size),
            nn.Softmax(dim=-1)
        )
        
        # Macro action value estimator
        self.macro_value_estimator = nn.Sequential(
            nn.Linear(128 + d_model, 64),  # sequence features + context
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Current macro execution state
        self.executing_macro = False
        self.current_macro_sequence = []
        self.current_macro_step = 0
        self.macro_start_reward = 0.0
        
        # Learning statistics
        self.macro_discovery_stats = {
            'patterns_discovered': 0,
            'macros_created': 0,
            'successful_executions': 0,
            'failed_executions': 0
        }
        
    def add_action(self, action, reward, state_context=None):
        """
        Add an action to the sequence for pattern detection.
        
        Args:
            action: Primitive action taken
            reward: Reward received after action
            state_context: Optional state context for better pattern recognition
        """
        self.recent_actions.append(action)
        
        # Update macro success tracking if executing
        if self.executing_macro:
            self._update_macro_execution(reward)
        
        # Detect patterns in recent actions
        self._detect_patterns()
        
        # Learn from successful sequences
        if reward > 0.5:  # Threshold for "successful" action
            self._learn_from_success()
            
    def _detect_patterns(self):
        """Detect repetitive patterns in recent action sequences."""
        if len(self.recent_actions) < self.max_sequence_length:
            return
            
        # Look for patterns of different lengths
        for seq_len in range(2, min(self.max_sequence_length + 1, len(self.recent_actions))):
            # Extract potential pattern
            pattern = tuple(list(self.recent_actions)[-seq_len:])
            
            # Skip patterns with too much repetition (like staying still)
            if len(set(pattern)) == 1:
                continue
                
            self.pattern_candidates[pattern] += 1
            
            # Create macro if pattern is frequent enough
            if (self.pattern_candidates[pattern] >= self.min_pattern_frequency and 
                pattern not in self.macro_vocab):
                self._create_macro_action(pattern)
                
    def _create_macro_action(self, pattern):
        """Create a new macro action from a detected pattern."""
        macro_id = self.next_macro_id
        self.next_macro_id += 1
        
        self.macro_vocab[pattern] = macro_id
        self.macro_sequences[macro_id] = list(pattern)
        self.macro_success_rates[macro_id] = 0.5  # Initialize with neutral success rate
        
        self.macro_discovery_stats['patterns_discovered'] += 1
        self.macro_discovery_stats['macros_created'] += 1
        
        print(f"Created macro action {macro_id}: {pattern}")
        
    def _learn_from_success(self):
        """Learn from successful action sequences."""
        if len(self.recent_actions) >= 2:
            # Reinforce recent successful patterns
            for seq_len in range(2, min(len(self.recent_actions) + 1, self.max_sequence_length + 1)):
                pattern = tuple(list(self.recent_actions)[-seq_len:])
                if pattern in self.macro_vocab:
                    macro_id = self.macro_vocab[pattern]
                    # Boost success rate slightly
                    current_rate = self.macro_success_rates[macro_id]
                    self.macro_success_rates[macro_id] = min(1.0, current_rate + 0.1)
                    
    def get_macro_action_probs(self, context_features, available_actions=None):
        """
        Get probability distribution over macro actions given context.
        
        Args:
            context_features: State context features [d_model]
            available_actions: List of available action IDs (None for all)
            
        Returns:
            dict: {action_id: probability} for both primitive and macro actions
        """
        action_probs = {}
        
        # Get primitive action probabilities (uniform for now, could be learned)
        primitive_prob = 0.7  # Reserve 70% probability mass for primitives
        for action_id in range(self.action_space_size):
            if available_actions is None or action_id in available_actions:
                action_probs[action_id] = primitive_prob / self.action_space_size
                
        # Get macro action probabilities
        if self.macro_sequences:
            macro_prob = 0.3  # Reserve 30% for macro actions
            total_macro_value = 0.0
            macro_values = {}
            
            # Compute value for each macro action
            for macro_id, sequence in self.macro_sequences.items():
                if available_actions is None or macro_id in available_actions:
                    # Encode sequence
                    seq_tensor = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        encoded, _ = self.sequence_encoder(seq_tensor)
                        sequence_features = encoded[:, -1, :]  # Use last hidden state
                        
                        # Combine with context
                        combined_features = torch.cat([sequence_features, context_features.unsqueeze(0)], dim=1)
                        macro_value = self.macro_value_estimator(combined_features).item()
                        
                    # Weight by success rate and usage
                    success_weight = self.macro_success_rates[macro_id]
                    usage_weight = min(1.0, self.macro_usage_counts[macro_id] / 10.0)  # Cap usage weight
                    
                    final_value = macro_value * success_weight * (0.5 + 0.5 * usage_weight)
                    macro_values[macro_id] = max(0.01, final_value)  # Minimum probability
                    total_macro_value += macro_values[macro_id]
                    
            # Normalize macro probabilities
            if total_macro_value > 0:
                for macro_id, value in macro_values.items():
                    action_probs[macro_id] = (value / total_macro_value) * macro_prob
                    
        return action_probs
        
    def execute_macro_action(self, macro_id):
        """
        Start executing a macro action.
        
        Args:
            macro_id: ID of macro action to execute
            
        Returns:
            int: First primitive action in the sequence
        """
        if macro_id not in self.macro_sequences:
            raise ValueError(f"Unknown macro action ID: {macro_id}")
            
        self.executing_macro = True
        self.current_macro_sequence = self.macro_sequences[macro_id].copy()
        self.current_macro_step = 0
        self.macro_usage_counts[macro_id] += 1
        
        # Return first action in sequence
        return self._get_next_macro_action()
        
    def _get_next_macro_action(self):
        """Get the next action in the current macro sequence."""
        if not self.executing_macro or self.current_macro_step >= len(self.current_macro_sequence):
            self.executing_macro = False
            return None
            
        action = self.current_macro_sequence[self.current_macro_step]
        self.current_macro_step += 1
        
        # Check if macro is complete
        if self.current_macro_step >= len(self.current_macro_sequence):
            self.executing_macro = False
            
        return action
        
    def _update_macro_execution(self, reward):
        """Update macro execution tracking with reward feedback."""
        if reward > 0.1:  # Positive reward
            self.macro_discovery_stats['successful_executions'] += 1
        elif reward < -0.1:  # Negative reward
            self.macro_discovery_stats['failed_executions'] += 1
            
    def is_executing_macro(self):
        """Check if currently executing a macro action."""
        return self.executing_macro
        
    def get_next_primitive_action(self):
        """Get next primitive action if executing macro, None otherwise."""
        return self._get_next_macro_action() if self.executing_macro else None
        
    def get_macro_stats(self):
        """Get statistics about macro action learning and usage."""
        stats = self.macro_discovery_stats.copy()
        stats.update({
            'total_macros': len(self.macro_sequences),
            'avg_success_rate': np.mean(list(self.macro_success_rates.values())) if self.macro_success_rates else 0.0,
            'most_used_macro': max(self.macro_usage_counts.items(), key=lambda x: x[1]) if self.macro_usage_counts else None,
            'pattern_candidates': len(self.pattern_candidates)
        })
        return stats
        
    def prune_unsuccessful_macros(self, min_usage=5, min_success_rate=0.3):
        """Remove macro actions that are not useful."""
        macros_to_remove = []
        
        for macro_id in list(self.macro_sequences.keys()):
            usage_count = self.macro_usage_counts[macro_id]
            success_rate = self.macro_success_rates[macro_id]
            
            # Remove if insufficient usage or poor success rate
            if usage_count >= min_usage and success_rate < min_success_rate:
                macros_to_remove.append(macro_id)
                
        for macro_id in macros_to_remove:
            # Find the original pattern
            pattern = None
            for p, mid in self.macro_vocab.items():
                if mid == macro_id:
                    pattern = p
                    break
                    
            if pattern:
                del self.macro_vocab[pattern]
            del self.macro_sequences[macro_id]
            del self.macro_success_rates[macro_id]
            del self.macro_usage_counts[macro_id]
            
        if macros_to_remove:
            print(f"Pruned {len(macros_to_remove)} unsuccessful macro actions")
            
    def save_state(self, filepath):
        """Save macro action learning state."""
        state = {
            'macro_vocab': self.macro_vocab,
            'macro_sequences': self.macro_sequences,
            'macro_success_rates': dict(self.macro_success_rates),
            'macro_usage_counts': dict(self.macro_usage_counts),
            'next_macro_id': self.next_macro_id,
            'pattern_candidates': dict(self.pattern_candidates),
            'discovery_stats': self.macro_discovery_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
        # Save neural network state
        torch.save(self.state_dict(), filepath.replace('.pkl', '_model.pth'))
        
    def load_state(self, filepath):
        """Load macro action learning state."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            self.macro_vocab = state['macro_vocab']
            self.macro_sequences = state['macro_sequences']
            self.macro_success_rates = defaultdict(float, state['macro_success_rates'])
            self.macro_usage_counts = defaultdict(int, state['macro_usage_counts'])
            self.next_macro_id = state['next_macro_id']
            self.pattern_candidates = defaultdict(int, state['pattern_candidates'])
            self.macro_discovery_stats = state['discovery_stats']
            
            # Load neural network state
            model_path = filepath.replace('.pkl', '_model.pth')
            self.load_state_dict(torch.load(model_path, map_location=self.device))
            
        except FileNotFoundError:
            print(f"Warning: Could not load macro action state from {filepath}")
            
    def reset_episode(self):
        """Reset episode-specific state."""
        self.executing_macro = False
        self.current_macro_sequence = []
        self.current_macro_step = 0
        self.macro_start_reward = 0.0