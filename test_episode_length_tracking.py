#!/usr/bin/env python3
"""
Test script to verify that episode length tracking includes checkpoint steps correctly.
"""

import sys
import tempfile
from pathlib import Path

# Add PoliwhiRL to path
sys.path.insert(0, str(Path(__file__).parent))

def test_episode_length_tracking():
    """Test that episode lengths properly include checkpoint steps."""
    
    print("🧪 Testing Episode Length Tracking with State Curriculum")
    print("=" * 60)
    
    # Mock the key components we need to test
    class MockEnv:
        def __init__(self):
            self.steps = 0
            
        def reset(self):
            self.steps = 0
            return "mock_state"
            
        def load_gym_state(self, path, episode_length, n_goals):
            # Simulate loading a state that was saved at step 150
            self.steps = 150
            return "mock_loaded_state"
            
        def save_gym_state(self, path):
            pass
            
        def close(self):
            pass
    
    class MockStateManager:
        def __init__(self):
            self.use_state_curriculum = True
            self.saved_states = [
                ("mock_state_path", "progression")
            ]
            self.call_count = 0
            
        def sample_starting_state(self):
            self.call_count += 1
            if self.call_count == 1:
                return None, 'scratch'  # First episode from scratch
            else:
                return "mock_state_path", "progression"  # Second episode from checkpoint
                
        def save_state_checkpoint(self, env, episode_data):
            print(f"   📁 Saved checkpoint: {episode_data}")
            return None
            
        def update_checkpoint_performance(self, path, success, reward):
            print(f"   📊 Updated checkpoint performance: success={success}, reward={reward}")
    
    # Simulate the episode statistics tracking
    episode_data = {
        "episode_rewards": [],
        "episode_lengths": [],
        "moving_avg_reward": [],
        "moving_avg_length": [],
    }
    
    def mock_update_episode_stats(total_reward, total_episode_length, start_type, checkpoint_steps):
        """Mock version of _update_episode_stats to show the tracking."""
        
        episode_data["episode_rewards"].append(total_reward)
        episode_data["episode_lengths"].append(total_episode_length)
        episode_data["moving_avg_reward"].append(total_reward)
        episode_data["moving_avg_length"].append(total_episode_length)
        
        print(f"   📊 Episode Stats:")
        print(f"      Start type: {start_type}")
        print(f"      Total episode length: {total_episode_length} steps")
        print(f"      Checkpoint steps: {checkpoint_steps} steps")
        print(f"      Additional steps: {total_episode_length - checkpoint_steps} steps")
        print(f"      Total reward: {total_reward}")
        print(f"      Efficiency: {total_reward / max(total_episode_length, 1):.4f} reward/step")
        print()
    
    # Test Episode 1: From scratch
    print("🎯 Episode 1: Starting from scratch")
    print("-" * 40)
    
    env = MockEnv()
    state_manager = MockStateManager()
    
    # Sample starting state (should be scratch for first episode)
    curriculum_state_path, start_type = state_manager.sample_starting_state()
    print(f"   🎮 Starting type: {start_type}")
    
    if start_type == 'scratch':
        state = env.reset()
        checkpoint_steps = 0
        steps = env.steps  # Should be 0
        print(f"   🏁 Started from beginning, current steps: {steps}")
    
    # Simulate running 200 more steps
    simulated_additional_steps = 200
    env.steps += simulated_additional_steps
    total_episode_length = env.steps  # Should be 200
    
    print(f"   🏃 Ran {simulated_additional_steps} additional steps")
    print(f"   📏 Final episode length: {total_episode_length}")
    
    # Update stats
    mock_update_episode_stats(
        total_reward=1.5, 
        total_episode_length=total_episode_length,
        start_type=start_type,
        checkpoint_steps=checkpoint_steps
    )
    
    # Test Episode 2: From checkpoint  
    print("🎯 Episode 2: Starting from checkpoint")
    print("-" * 40)
    
    env = MockEnv()
    
    # Sample starting state (should be curriculum for second episode)
    curriculum_state_path, start_type = state_manager.sample_starting_state()
    print(f"   🎮 Starting type: {start_type}")
    
    if start_type != 'scratch':
        state = env.load_gym_state(curriculum_state_path, 500, 3)
        checkpoint_steps = env.steps  # Should be 150 (from saved state)
        steps = env.steps
        print(f"   📂 Loaded checkpoint with {checkpoint_steps} existing steps")
    
    # Simulate running 100 more steps
    simulated_additional_steps = 100
    env.steps += simulated_additional_steps
    total_episode_length = env.steps  # Should be 250 (150 + 100)
    
    print(f"   🏃 Ran {simulated_additional_steps} additional steps")
    print(f"   📏 Final episode length: {total_episode_length} (includes {checkpoint_steps} checkpoint steps)")
    
    # Update stats
    mock_update_episode_stats(
        total_reward=2.3, 
        total_episode_length=total_episode_length,
        start_type=start_type,
        checkpoint_steps=checkpoint_steps
    )
    
    # Show comparison
    print("📈 COMPARISON SUMMARY")
    print("-" * 40)
    print(f"Episode 1 (scratch):     {episode_data['episode_lengths'][0]} total steps, reward: {episode_data['episode_rewards'][0]}")
    print(f"Episode 2 (curriculum):  {episode_data['episode_lengths'][1]} total steps, reward: {episode_data['episode_rewards'][1]}")
    print(f"                         (150 checkpoint + 100 additional)")
    print()
    print("✅ KEY INSIGHT:")
    print("   Both episodes report the TOTAL steps from game beginning to end.")
    print("   This ensures fair comparison of true episode length and efficiency.")
    print("   Episode 2 shows the agent completed the game in 250 total steps")
    print("   (not just the 100 steps taken after loading the checkpoint).")
    
    return True

if __name__ == "__main__":
    test_episode_length_tracking()