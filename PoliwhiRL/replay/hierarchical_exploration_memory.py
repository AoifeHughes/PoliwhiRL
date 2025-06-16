# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque
import hashlib
import pickle


class HierarchicalExplorationMemory:
    """
    Enhanced exploration memory with hierarchical attention and event detection.
    
    Replaces simple pooling with learned attention mechanisms for better
    spatial-temporal understanding of important locations and events.
    """
    
    def __init__(self, max_size=100, history_length=5, use_memory=True, action_space_size=9, device='cpu'):
        """
        Initialize hierarchical exploration memory.
        
        Args:
            max_size: Maximum number of recent screen hashes to store
            history_length: Number of past screen hashes for sequence modeling
            use_memory: Whether to use the memory functionality
            action_space_size: Number of possible actions for action context
            device: Device for neural computations
        """
        self.max_size = max_size
        self.history_length = history_length
        self.use_memory = use_memory
        self.action_space_size = action_space_size
        self.device = torch.device(device)
        
        # Core memory storage
        self.memory = []  # List of (hash, location, action_seq, importance, timestamp) tuples
        self.hash_visits = defaultdict(int)
        self.recent_hashes = deque(maxlen=history_length)
        
        # Event detection
        self.important_events = []  # Store significant events (goals, map changes, etc.)
        self.last_location = None
        self.last_map_id = None
        self.step_count = 0
        
        # Hierarchical memory components
        self.waypoints = []  # Important spatial locations
        self.map_transitions = []  # Map change events
        self.goal_achievements = []  # Goal completion events
        
        # Attention mechanism for memory importance
        self.importance_network = self._build_importance_network()
        
        # Cached tensors for performance
        self._cached_tensor = None
        self._cache_valid = False
        
    def _build_importance_network(self):
        """Build neural network to compute memory importance scores."""
        return nn.Sequential(
            nn.Linear(8, 32),  # location_xy + visit_count + action_context + recency + event_flags
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Importance score 0-1
        ).to(self.device)
        
    def add_screen(self, screen_array):
        """Add a screen for backward compatibility (calls add_transition with dummy values)."""
        if not self.use_memory:
            return
        # Call add_transition with default values for backward compatibility
        self.add_transition(screen_array, 0, screen_array, 0.0, None)
    
    def add_transition(self, state, action, next_state, reward, coordinates=None):
        """
        Add a transition with enhanced event detection.
        
        Args:
            state: Current game state/screen
            action: Action taken
            next_state: Resulting state
            reward: Reward received
            coordinates: Optional dict with x, y, map_num info
        """
        if not self.use_memory:
            return
            
        self.step_count += 1
        
        # Compute state hash
        state_hash = self._compute_hash(state)
        
        # Extract location information
        location = self._extract_location(coordinates, state)
        
        # Detect important events
        events = self._detect_events(location, reward, action)
        
        # Update visit counts
        self.hash_visits[state_hash] += 1
        
        # Update recent hashes for sequence modeling
        self.recent_hashes.append(state_hash)
        
        # Compute importance score
        importance = self._compute_importance(location, action, events, reward)
        
        # Store in memory with metadata
        memory_entry = {
            'hash': state_hash,
            'location': location,
            'action': action,
            'importance': importance,
            'timestamp': self.step_count,
            'events': events,
            'reward': reward,
            'visit_count': self.hash_visits[state_hash]
        }
        
        # Add to memory
        self._add_to_memory(memory_entry)
        
        # Update cached structures
        self._update_hierarchical_structures(memory_entry)
        
        # Invalidate tensor cache
        self._cache_valid = False
        
    def _extract_location(self, coordinates, state):
        """Extract location information from coordinates or state."""
        if coordinates:
            # Handle both tuple format (x, y, map_num) and dict format
            if isinstance(coordinates, (tuple, list)) and len(coordinates) >= 3:
                return {
                    'x': coordinates[0] or 0,
                    'y': coordinates[1] or 0, 
                    'map_num': coordinates[2] or 0,
                    'room': 0
                }
            elif isinstance(coordinates, dict):
                return {
                    'x': coordinates.get('x', 0),
                    'y': coordinates.get('y', 0),
                    'map_num': coordinates.get('map_num', 0),
                    'room': coordinates.get('room', 0)
                }
        
        # Fallback to basic hash-based location
        return {
            'x': 0, 'y': 0, 'map_num': 0, 'room': 0
        }
    
    def _detect_events(self, location, reward, action):
        """Detect important events for hierarchical memory."""
        events = []
        
        # Goal achievement detection (high reward)
        if reward > 1.0:
            events.append('goal_achievement')
            
        # Map transition detection
        if self.last_location and location['map_num'] != self.last_location.get('map_num', 0):
            events.append('map_transition')
            
        # First visit to location
        location_key = (location['x'], location['y'], location['map_num'])
        if location_key not in [loc for loc in getattr(self, '_visited_locations', set())]:
            events.append('new_location')
            if not hasattr(self, '_visited_locations'):
                self._visited_locations = set()
            self._visited_locations.add(location_key)
            
        # Stuck detection (repeated actions in same location)
        if (len(self.recent_hashes) >= 3 and 
            all(h == self.recent_hashes[-1] for h in list(self.recent_hashes)[-3:])):
            events.append('stuck_behavior')
            
        self.last_location = location
        return events
        
    def _compute_importance(self, location, action, events, reward):
        """Compute importance score using neural network."""
        # Create feature vector for importance computation
        features = torch.tensor([
            location['x'] / 255.0,  # Normalized position
            location['y'] / 255.0,
            location['map_num'] / 255.0,
            action / self.action_space_size,  # Normalized action
            min(reward, 10.0) / 10.0,  # Clamped normalized reward
            len(events),  # Number of events
            1.0 if 'goal_achievement' in events else 0.0,
            1.0 if 'map_transition' in events else 0.0
        ], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            importance = self.importance_network(features).item()
            
        return importance
        
    def _add_to_memory(self, memory_entry):
        """Add entry to memory with size management."""
        # Check if similar entry exists
        for i, existing in enumerate(self.memory):
            if existing['hash'] == memory_entry['hash']:
                # Update existing entry with higher importance
                if memory_entry['importance'] > existing['importance']:
                    self.memory[i] = memory_entry
                return
                
        # Add new entry
        self.memory.append(memory_entry)
        
        # Remove oldest low-importance entries if over capacity
        if len(self.memory) > self.max_size:
            # Sort by importance and recency, remove least important old entries
            self.memory.sort(key=lambda x: (x['importance'], x['timestamp']), reverse=True)
            self.memory = self.memory[:self.max_size]
            
    def _update_hierarchical_structures(self, memory_entry):
        """Update waypoints and event lists."""
        # Add to waypoints if high importance
        if memory_entry['importance'] > 0.7:
            waypoint = {
                'location': memory_entry['location'],
                'importance': memory_entry['importance'],
                'timestamp': memory_entry['timestamp']
            }
            self.waypoints.append(waypoint)
            
            # Keep only top waypoints
            if len(self.waypoints) > 20:
                self.waypoints.sort(key=lambda x: x['importance'], reverse=True)
                self.waypoints = self.waypoints[:20]
                
        # Store important events
        if 'map_transition' in memory_entry['events']:
            self.map_transitions.append(memory_entry)
            
        if 'goal_achievement' in memory_entry['events']:
            self.goal_achievements.append(memory_entry)
            
    def get_memory_tensor(self):
        """
        Generate memory tensor with hierarchical attention.
        
        Returns:
            numpy array: Enhanced memory representation
        """
        if not self.use_memory or not self.memory:
            return np.zeros((self.max_size, 8), dtype=np.float32)  # Expanded feature size
            
        # Use cached tensor if valid
        if self._cache_valid and self._cached_tensor is not None:
            return self._cached_tensor
            
        # Create enhanced tensor with more features
        tensor = np.zeros((self.max_size, 8), dtype=np.float32)
        
        # Sort memory by importance and recency
        sorted_memory = sorted(self.memory, key=lambda x: (x['importance'], x['timestamp']), reverse=True)
        
        for i, entry in enumerate(sorted_memory[:self.max_size]):
            # Enhanced features
            tensor[i, 0] = entry['visit_count']  # Visit frequency
            tensor[i, 1] = entry['importance']  # Computed importance
            tensor[i, 2] = (self.step_count - entry['timestamp']) / max(self.step_count, 1)  # Recency
            tensor[i, 3] = entry['location']['x'] / 255.0  # Normalized x
            tensor[i, 4] = entry['location']['y'] / 255.0  # Normalized y
            tensor[i, 5] = entry['location']['map_num'] / 255.0  # Normalized map
            tensor[i, 6] = 1.0 if 'goal_achievement' in entry['events'] else 0.0  # Goal flag
            tensor[i, 7] = 1.0 if 'map_transition' in entry['events'] else 0.0  # Transition flag
            
        # Cache the tensor
        self._cached_tensor = tensor
        self._cache_valid = True
        
        return tensor
        
    def get_waypoint_summary(self):
        """Get summary of important waypoints for debugging."""
        return {
            'total_waypoints': len(self.waypoints),
            'map_transitions': len(self.map_transitions),
            'goal_achievements': len(self.goal_achievements),
            'total_locations': len(self._visited_locations) if hasattr(self, '_visited_locations') else 0,
            'memory_utilization': len(self.memory) / self.max_size
        }
        
    def _compute_hash(self, screen_array):
        """Compute hash of screen array."""
        array_bytes = screen_array.tobytes()
        return hashlib.md5(array_bytes).hexdigest()
        
    def reset(self):
        """Reset the memory bank."""
        self.memory = []
        self.hash_visits = defaultdict(int)
        self.recent_hashes.clear()
        self.important_events = []
        self.waypoints = []
        self.map_transitions = []
        self.goal_achievements = []
        self.last_location = None
        self.last_map_id = None
        self.step_count = 0
        self._cache_valid = False
        if hasattr(self, '_visited_locations'):
            self._visited_locations.clear()
            
    def save(self, filepath):
        """Save memory state to file."""
        state = {
            'memory': self.memory,
            'hash_visits': dict(self.hash_visits),
            'waypoints': self.waypoints,
            'map_transitions': self.map_transitions,
            'goal_achievements': self.goal_achievements,
            'step_count': self.step_count,
            'visited_locations': getattr(self, '_visited_locations', set())
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    def load(self, filepath):
        """Load memory state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.memory = state['memory']
            self.hash_visits = defaultdict(int, state['hash_visits'])
            self.waypoints = state['waypoints']
            self.map_transitions = state['map_transitions']
            self.goal_achievements = state['goal_achievements']
            self.step_count = state['step_count']
            self._visited_locations = state.get('visited_locations', set())
            self._cache_valid = False
            
        except FileNotFoundError:
            print(f"Warning: Could not load exploration memory from {filepath}")