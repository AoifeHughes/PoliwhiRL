# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import random


@dataclass
class StateCheckpoint:
    """Represents a saved state checkpoint with metadata."""
    state_path: str                  # Path to saved gym state
    category: str                   # foundation, progression, frontier, goal
    difficulty: float               # 0.0 (easy) to 1.0 (hard)
    goal_progress: float           # 0.0 (no progress) to 1.0 (goal achieved)
    success_rate: float            # Success rate when starting from this state
    reward_potential: float        # Average reward potential from this state
    exploration_value: float       # How much exploration this state enables
    timestamp: float               # When this state was saved
    episode_number: int           # Episode when state was saved
    curriculum_stage: int         # Which curriculum stage this belongs to
    metadata: Dict[str, Any]      # Additional state-specific information
    usage_count: int = 0          # How many times this state has been used
    recent_success_rate: float = 0.0  # Recent success rate (last 10 uses)


class CurriculumStateManager:
    """
    Manages checkpoint states for adaptive curriculum learning.
    
    Maintains a buffer of states at different difficulty levels and provides
    intelligent state selection to focus training on challenging areas while
    preventing catastrophic forgetting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_state_curriculum = config.get("use_state_curriculum", False)
        self.state_buffer_size = config.get("state_buffer_size", 100)
        # Make save directory stage-specific for curriculum learning
        base_save_dir = config.get("state_save_directory", "./curriculum_states")
        output_base_dir = config.get("output_base_dir", "")
        self.current_stage = config.get("N_goals_target", 1)
        
        if output_base_dir:
            # If we have a stage-specific output directory, use it
            self.save_directory = os.path.join(output_base_dir, "curriculum_states")
        else:
            # Use stage-specific subdirectory
            stage_suffix = f"_stage_{self.current_stage}"
            self.save_directory = f"{base_save_dir}{stage_suffix}"
        
        # State buffer organized by category
        self.state_buffer: Dict[str, List[StateCheckpoint]] = {
            'foundation': [],    # Early game, well-learned states
            'progression': [],   # Mid-curriculum, moderate difficulty
            'frontier': [],      # Current learning edge, challenging
            'goal': []          # Near/at goal completion
        }
        
        # Performance tracking
        self.category_success_rates = defaultdict(lambda: deque(maxlen=50))
        self.validation_results = deque(maxlen=20)
        self.usage_statistics = defaultdict(int)
        
        # Sampling configuration
        self.base_distributions = {
            'early_stage': {'scratch': 0.7, 'foundation': 0.2, 'progression': 0.1, 'frontier': 0.0, 'goal': 0.0},
            'mid_stage': {'scratch': 0.4, 'foundation': 0.3, 'progression': 0.3, 'frontier': 0.0, 'goal': 0.0},
            'late_stage': {'scratch': 0.1, 'foundation': 0.2, 'progression': 0.4, 'frontier': 0.2, 'goal': 0.1},
            'advanced_stage': {'scratch': 0.1, 'foundation': 0.1, 'progression': 0.3, 'frontier': 0.4, 'goal': 0.1}
        }
        
        # Current sampling distribution
        self.current_distribution = self.base_distributions['early_stage'].copy()
        
        # Performance monitoring
        self.catastrophic_forgetting_threshold = config.get("catastrophic_forgetting_threshold", 0.8)
        self.validation_frequency = config.get("validation_frequency", 50)
        self.episodes_since_validation = 0
        
        # Safety monitoring
        self.performance_history = deque(maxlen=100)  # Track recent performance
        self.baseline_performance = None
        self.forgetting_detected = False
        self.safety_override_active = False
        self.last_safety_check = 0
        
        # Create save directory
        os.makedirs(self.save_directory, exist_ok=True)
        self.load_state_buffer()
        
    def save_state_checkpoint(self, env, episode_data: Dict[str, Any]) -> Optional[StateCheckpoint]:
        """
        Save a state checkpoint with appropriate categorization.
        
        Args:
            env: Environment instance with state saving capability
            episode_data: Dictionary containing episode information
            
        Returns:
            StateCheckpoint if saved, None otherwise
        """
        if not self.use_state_curriculum:
            return None
            
        # Determine if this state is worth saving
        if not self._should_save_state(episode_data):
            return None
            
        # Create unique state filename
        timestamp = time.time()
        state_filename = f"state_ep{episode_data.get('episode', 0)}_stage{self.current_stage}_{int(timestamp)}.pkl"
        state_path = os.path.join(self.save_directory, state_filename)
        
        # Save the gym state
        try:
            env.save_gym_state(state_path)
        except Exception as e:
            print(f"Failed to save state: {e}")
            return None
            
        # Create checkpoint metadata
        checkpoint = StateCheckpoint(
            state_path=state_path,
            category=self._categorize_state(episode_data),
            difficulty=self._calculate_difficulty(episode_data),
            goal_progress=self._calculate_goal_progress(episode_data),
            success_rate=0.5,  # Initialize with neutral success rate
            reward_potential=episode_data.get('total_reward', 0.0),
            exploration_value=self._calculate_exploration_value(episode_data),
            timestamp=timestamp,
            episode_number=episode_data.get('episode', 0),
            curriculum_stage=self.current_stage,
            metadata=self._extract_metadata(episode_data)
        )
        
        # Add to appropriate buffer
        self._add_to_buffer(checkpoint)
        
        # Persist the buffer
        self.save_state_buffer()
        
        return checkpoint
        
    def sample_starting_state(self) -> Tuple[Optional[str], str]:
        """
        Sample a starting state based on current curriculum needs.
        
        Returns:
            Tuple of (state_path, state_type) where state_path is None for scratch start
        """
        if not self.use_state_curriculum or not any(self.state_buffer.values()):
            return None, 'scratch'
            
        # Update distribution based on performance
        self._update_sampling_distribution()
        
        # Sample state type
        state_type = self._sample_state_type()
        
        if state_type == 'scratch':
            return None, 'scratch'
            
        # Sample specific state from category
        checkpoint = self._sample_from_category(state_type)
        
        if checkpoint is None:
            return None, 'scratch'  # Fallback to scratch
            
        # Update usage statistics
        checkpoint.usage_count += 1
        self.usage_statistics[state_type] += 1
        
        return checkpoint.state_path, state_type
        
    def update_checkpoint_performance(self, state_path: str, success: bool, reward: float):
        """Update performance metrics for a specific checkpoint."""
        checkpoint = self._find_checkpoint_by_path(state_path)
        if checkpoint:
            # Update success rate with exponential moving average
            alpha = 0.1
            checkpoint.success_rate = (1 - alpha) * checkpoint.success_rate + alpha * (1.0 if success else 0.0)
            
            # Update recent success rate
            if not hasattr(checkpoint, '_recent_outcomes'):
                checkpoint._recent_outcomes = deque(maxlen=10)
            checkpoint._recent_outcomes.append(success)
            checkpoint.recent_success_rate = sum(checkpoint._recent_outcomes) / len(checkpoint._recent_outcomes)
            
            # Update category success rates for distribution adjustment
            self.category_success_rates[checkpoint.category].append(success)
            
            # Add to performance history for safety monitoring
            self.performance_history.append(reward)
            self._check_safety_conditions()
            
    def validate_foundation_skills(self, validation_episodes: int = 5) -> float:
        """
        Run validation episodes starting from scratch to check for catastrophic forgetting.
        
        Returns:
            Average success rate on foundation skills
        """
        # This would be called by the training loop to run validation episodes
        # For now, we'll track when validation should occur
        self.episodes_since_validation += 1
        
        # Return current foundation success rate as proxy
        if 'foundation' in self.category_success_rates and self.category_success_rates['foundation']:
            return sum(self.category_success_rates['foundation']) / len(self.category_success_rates['foundation'])
        return 1.0  # Assume good performance if no data
        
    def update_curriculum_stage(self, new_stage: int):
        """Update curriculum stage and adapt state management accordingly."""
        old_stage = self.current_stage
        self.current_stage = new_stage
        
        # Update save directory for new stage
        if new_stage != old_stage:
            old_save_directory = self.save_directory
            
            # Recalculate save directory for new stage
            base_save_dir = self.config.get("state_save_directory", "./curriculum_states")
            output_base_dir = self.config.get("output_base_dir", "")
            
            if output_base_dir:
                self.save_directory = os.path.join(output_base_dir, "curriculum_states")
            else:
                stage_suffix = f"_stage_{new_stage}"
                self.save_directory = f"{base_save_dir}{stage_suffix}"
            
            # Create new save directory
            os.makedirs(self.save_directory, exist_ok=True)
            
            # Save current buffer to old location before transition
            if old_stage > 0 and old_save_directory != self.save_directory:
                old_buffer_path = os.path.join(old_save_directory, "state_buffer.pkl")
                if os.path.exists(old_buffer_path):
                    print(f"📁 Preserving stage {old_stage} state buffer at {old_buffer_path}")
                    
        # Update sampling distribution based on new stage
        stage_map = {
            1: 'early_stage',
            2: 'early_stage', 
            3: 'mid_stage',
            4: 'mid_stage',
            5: 'late_stage',
            6: 'late_stage',
            7: 'advanced_stage'
        }
        
        stage_key = stage_map.get(new_stage, 'advanced_stage')
        self.current_distribution = self.base_distributions[stage_key].copy()
        
        # Clean up old states if stage changed significantly
        if new_stage > old_stage + 1:
            self._cleanup_old_states(old_stage)
            
        # Load state buffer for new stage
        if new_stage != old_stage:
            self.load_state_buffer()
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about state management."""
        stats = {
            'total_states': sum(len(states) for states in self.state_buffer.values()),
            'states_by_category': {cat: len(states) for cat, states in self.state_buffer.items()},
            'current_distribution': self.current_distribution.copy(),
            'usage_statistics': dict(self.usage_statistics),
            'average_success_rates': {}
        }
        
        # Calculate average success rates by category
        for category, success_list in self.category_success_rates.items():
            if success_list:
                stats['average_success_rates'][category] = sum(success_list) / len(success_list)
        
        # Add safety monitoring status
        stats['safety_status'] = self.get_safety_status()
                
        return stats
        
    def _should_save_state(self, episode_data: Dict[str, Any]) -> bool:
        """Determine if a state is worth saving."""
        # Save if high reward
        if episode_data.get('total_reward', 0) > 1.0:
            return True
            
        # Save if goal progress made
        if episode_data.get('goals_achieved', 0) > 0:
            return True
            
        # Save if significant exploration
        if episode_data.get('new_locations', 0) > 5:
            return True
            
        # Save if near goal (high steps with some reward)
        if episode_data.get('steps', 0) > self.config.get('episode_length', 500) * 0.8:
            if episode_data.get('total_reward', 0) > 0.5:
                return True
                
        # Random sampling for diversity
        if random.random() < 0.1:  # 10% chance
            return True
            
        return False
        
    def _categorize_state(self, episode_data: Dict[str, Any]) -> str:
        """Categorize state based on progress and difficulty."""
        goals_achieved = episode_data.get('goals_achieved', 0)
        total_goals = self.config.get('N_goals_target', 1)
        steps = episode_data.get('steps', 0)
        max_steps = self.config.get('episode_length', 500)
        
        progress_ratio = goals_achieved / max(total_goals, 1)
        step_ratio = steps / max_steps
        
        # Goal states: near or at goal completion
        if goals_achieved >= total_goals or (goals_achieved > 0 and step_ratio > 0.7):
            return 'goal'
            
        # Frontier states: challenging, making progress but not completing
        elif progress_ratio > 0.3 and step_ratio > 0.5:
            return 'frontier'
            
        # Progression states: moderate progress
        elif progress_ratio > 0.1 or step_ratio > 0.3:
            return 'progression'
            
        # Foundation states: early game
        else:
            return 'foundation'
            
    def _calculate_difficulty(self, episode_data: Dict[str, Any]) -> float:
        """Calculate difficulty score for the state."""
        goals_achieved = episode_data.get('goals_achieved', 0)
        total_goals = self.config.get('N_goals_target', 1)
        steps = episode_data.get('steps', 0)
        max_steps = self.config.get('episode_length', 500)
        
        # Higher difficulty for states with more progress
        progress_difficulty = goals_achieved / max(total_goals, 1)
        step_difficulty = steps / max_steps
        
        return min(1.0, (progress_difficulty + step_difficulty) / 2.0)
        
    def _calculate_goal_progress(self, episode_data: Dict[str, Any]) -> float:
        """Calculate goal progress ratio."""
        goals_achieved = episode_data.get('goals_achieved', 0)
        total_goals = self.config.get('N_goals_target', 1)
        return goals_achieved / max(total_goals, 1)
        
    def _calculate_exploration_value(self, episode_data: Dict[str, Any]) -> float:
        """Calculate exploration value of the state."""
        new_locations = episode_data.get('new_locations', 0)
        total_locations = episode_data.get('total_locations', 1)
        
        # Higher value for states that enable more exploration
        return min(1.0, new_locations / max(total_locations, 1))
        
    def _extract_metadata(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metadata from episode data."""
        return {
            'total_reward': episode_data.get('total_reward', 0.0),
            'steps': episode_data.get('steps', 0),
            'goals_achieved': episode_data.get('goals_achieved', 0),
            'new_locations': episode_data.get('new_locations', 0),
            'curriculum_stage': self.current_stage
        }
        
    def _add_to_buffer(self, checkpoint: StateCheckpoint):
        """Add checkpoint to appropriate buffer with size management."""
        category = checkpoint.category
        self.state_buffer[category].append(checkpoint)
        
        # Maintain buffer size limits
        max_per_category = self.state_buffer_size // 4
        if len(self.state_buffer[category]) > max_per_category:
            # Remove oldest, lowest success rate states
            self.state_buffer[category].sort(key=lambda x: (x.success_rate, x.timestamp))
            self.state_buffer[category] = self.state_buffer[category][-max_per_category:]
            
    def _update_sampling_distribution(self):
        """Update sampling distribution based on recent performance."""
        # Check for catastrophic forgetting
        foundation_success = self.validate_foundation_skills()
        
        if foundation_success < self.catastrophic_forgetting_threshold:
            # Increase scratch and foundation probability
            self.current_distribution['scratch'] = min(0.8, self.current_distribution['scratch'] + 0.2)
            self.current_distribution['foundation'] = min(0.5, self.current_distribution['foundation'] + 0.1)
            # Reduce other probabilities proportionally
            total_others = sum(self.current_distribution[k] for k in ['progression', 'frontier', 'goal'])
            if total_others > 0:
                reduction_factor = (1.0 - self.current_distribution['scratch'] - self.current_distribution['foundation']) / total_others
                for k in ['progression', 'frontier', 'goal']:
                    self.current_distribution[k] *= reduction_factor
                    
    def _sample_state_type(self) -> str:
        """Sample a state type based on current distribution."""
        types = list(self.current_distribution.keys())
        probabilities = list(self.current_distribution.values())
        
        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / len(types)] * len(types)
            
        return np.random.choice(types, p=probabilities)
        
    def _sample_from_category(self, category: str) -> Optional[StateCheckpoint]:
        """Sample a specific checkpoint from a category."""
        if category not in self.state_buffer or not self.state_buffer[category]:
            return None
            
        # Weight by success rate and exploration value
        states = self.state_buffer[category]
        weights = []
        
        for state in states:
            # Prefer states with moderate success rates (not too easy, not too hard)
            success_weight = 1.0 - abs(state.success_rate - 0.6)  # Peak at 60% success rate
            exploration_weight = state.exploration_value
            recency_weight = np.exp(-(time.time() - state.timestamp) / (24 * 3600))  # Decay over days
            
            weight = success_weight * 0.5 + exploration_weight * 0.3 + recency_weight * 0.2
            weights.append(max(0.1, weight))  # Minimum weight
            
        # Sample based on weights
        if not weights:
            return random.choice(states)
            
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        idx = np.random.choice(len(states), p=weights)
        return states[idx]
        
    def _find_checkpoint_by_path(self, state_path: str) -> Optional[StateCheckpoint]:
        """Find checkpoint by state path."""
        for category in self.state_buffer.values():
            for checkpoint in category:
                if checkpoint.state_path == state_path:
                    return checkpoint
        return None
        
    def _cleanup_old_states(self, old_stage: int):
        """Clean up states from much older curriculum stages."""
        for category in self.state_buffer:
            self.state_buffer[category] = [
                state for state in self.state_buffer[category]
                if state.curriculum_stage >= old_stage - 1  # Keep states from at most 1 stage back
            ]
            
    def save_state_buffer(self):
        """Persist state buffer to disk."""
        buffer_path = os.path.join(self.save_directory, "state_buffer.pkl")
        try:
            # Convert to serializable format
            serializable_buffer = {}
            for category, states in self.state_buffer.items():
                serializable_buffer[category] = [asdict(state) for state in states]
                
            data = {
                'state_buffer': serializable_buffer,
                'category_success_rates': {k: list(v) for k, v in self.category_success_rates.items()},
                'usage_statistics': dict(self.usage_statistics),
                'current_distribution': self.current_distribution,
                'current_stage': self.current_stage
            }
            
            with open(buffer_path, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            print(f"Failed to save state buffer: {e}")
            
    def load_state_buffer(self):
        """Load state buffer from disk."""
        buffer_path = os.path.join(self.save_directory, "state_buffer.pkl")
        
        if not os.path.exists(buffer_path):
            return
            
        try:
            with open(buffer_path, 'rb') as f:
                data = pickle.load(f)
                
            # Reconstruct state buffer
            for category, states_data in data.get('state_buffer', {}).items():
                self.state_buffer[category] = [
                    StateCheckpoint(**state_data) for state_data in states_data
                    if os.path.exists(state_data.get('state_path', ''))  # Only load if state file exists
                ]
                
            # Restore other data
            for category, success_list in data.get('category_success_rates', {}).items():
                self.category_success_rates[category] = deque(success_list, maxlen=50)
                
            self.usage_statistics.update(data.get('usage_statistics', {}))
            self.current_distribution.update(data.get('current_distribution', {}))
            self.current_stage = data.get('current_stage', self.current_stage)
            
        except Exception as e:
            print(f"Failed to load state buffer: {e}")
            # Continue with empty buffer if loading fails
            
    def _check_safety_conditions(self):
        """Check for catastrophic forgetting and other safety issues."""
        if len(self.performance_history) < 20:
            return  # Need enough data
            
        # Set baseline after enough episodes
        if self.baseline_performance is None and len(self.performance_history) >= 50:
            self.baseline_performance = np.mean(list(self.performance_history)[:30])
            print(f"📊 Baseline performance established: {self.baseline_performance:.3f}")
            
        if self.baseline_performance is not None:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            performance_ratio = recent_performance / max(self.baseline_performance, 0.001)
            
            # Check for catastrophic forgetting
            if performance_ratio < self.catastrophic_forgetting_threshold:
                if not self.forgetting_detected:
                    self.forgetting_detected = True
                    print(f"⚠️ CATASTROPHIC FORGETTING DETECTED!")
                    print(f"   Recent performance: {recent_performance:.3f}")
                    print(f"   Baseline performance: {self.baseline_performance:.3f}")
                    print(f"   Performance ratio: {performance_ratio:.3f}")
                    self._activate_safety_override()
            else:
                # Recovery detected
                if self.forgetting_detected and performance_ratio > self.catastrophic_forgetting_threshold + 0.1:
                    self.forgetting_detected = False
                    self._deactivate_safety_override()
                    print(f"✅ Performance recovery detected - safety override deactivated")
                    
    def _activate_safety_override(self):
        """Activate safety measures to prevent further degradation."""
        self.safety_override_active = True
        
        # Increase scratch probability to 90% to focus on fundamentals
        self.current_distribution = {
            'scratch': 0.9,
            'foundation': 0.1,
            'progression': 0.0,
            'frontier': 0.0,
            'goal': 0.0
        }
        
        print(f"🚨 SAFETY OVERRIDE ACTIVATED:")
        print(f"   - Increased scratch training to 90%")
        print(f"   - Focusing on foundation skills")
        print(f"   - Advanced states temporarily disabled")
        
    def _deactivate_safety_override(self):
        """Deactivate safety override and return to normal curriculum."""
        self.safety_override_active = False
        
        # Restore normal distribution based on current stage
        stage_map = {
            1: 'early_stage',
            2: 'early_stage', 
            3: 'mid_stage',
            4: 'mid_stage',
            5: 'late_stage',
            6: 'late_stage',
            7: 'advanced_stage'
        }
        
        stage_key = stage_map.get(self.current_stage, 'advanced_stage')
        self.current_distribution = self.base_distributions[stage_key].copy()
        
        print(f"✅ Safety override deactivated - normal curriculum resumed")
        
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety monitoring status."""
        recent_perf = np.mean(list(self.performance_history)[-10:]) if len(self.performance_history) >= 10 else 0.0
        baseline_perf = self.baseline_performance or 0.0
        perf_ratio = recent_perf / max(baseline_perf, 0.001) if baseline_perf > 0 else 1.0
        
        return {
            'forgetting_detected': self.forgetting_detected,
            'safety_override_active': self.safety_override_active,
            'performance_history_length': len(self.performance_history),
            'recent_performance': recent_perf,
            'baseline_performance': baseline_perf,
            'performance_ratio': perf_ratio,
            'threshold': self.catastrophic_forgetting_threshold
        }