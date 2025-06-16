#!/usr/bin/env python3
"""
Test script for the State Curriculum system.
Tests the CurriculumStateManager integration without running full training.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add PoliwhiRL to path
sys.path.insert(0, str(Path(__file__).parent))

from PoliwhiRL.curriculum.state_manager import CurriculumStateManager


def test_state_manager():
    """Test basic state manager functionality."""
    print("🧪 Testing CurriculumStateManager...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            "use_state_curriculum": True,
            "state_buffer_size": 10,
            "state_save_directory": temp_dir,
            "N_goals_target": 1,
            "episode_length": 500,
            "catastrophic_forgetting_threshold": 0.8,
            "validation_frequency": 50
        }
        
        # Initialize state manager
        manager = CurriculumStateManager(config)
        print(f"✅ State manager initialized")
        print(f"   Buffer size: {manager.state_buffer_size}")
        print(f"   Save directory: {manager.save_directory}")
        print(f"   Current stage: {manager.current_stage}")
        
        # Test sampling with empty buffer (should return scratch)
        state_path, state_type = manager.sample_starting_state()
        print(f"✅ Empty buffer sampling: {state_type} (expected: scratch)")
        assert state_path is None
        assert state_type == 'scratch'
        
        # Test state saving (without actual environment)
        print("🧪 Testing state checkpoint saving...")
        
        # Mock episode data
        episode_data_list = [
            {
                'episode': 1,
                'total_reward': 2.5,
                'steps': 200,
                'goals_achieved': 1,
                'new_locations': 10,
                'total_locations': 15
            },
            {
                'episode': 2,
                'total_reward': 0.8,
                'steps': 450,
                'goals_achieved': 0,
                'new_locations': 5,
                'total_locations': 20
            },
            {
                'episode': 3,
                'total_reward': 1.2,
                'steps': 300,
                'goals_achieved': 1,
                'new_locations': 8,
                'total_locations': 28
            }
        ]
        
        # Mock environment class for testing
        class MockEnv:
            def save_gym_state(self, save_path):
                # Create a dummy state file
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(b"mock_state_data")
                print(f"   Mock state saved to: {save_path}")
        
        mock_env = MockEnv()
        
        # Test saving multiple states
        saved_checkpoints = []
        for episode_data in episode_data_list:
            checkpoint = manager.save_state_checkpoint(mock_env, episode_data)
            if checkpoint:
                saved_checkpoints.append(checkpoint)
                print(f"✅ Checkpoint saved: {checkpoint.category} (difficulty: {checkpoint.difficulty:.3f}, reward: {checkpoint.reward_potential:.2f})")
        
        print(f"✅ Saved {len(saved_checkpoints)} checkpoints")
        
        # Test sampling with populated buffer
        if saved_checkpoints:
            print("🧪 Testing sampling with populated buffer...")
            for i in range(5):
                state_path, state_type = manager.sample_starting_state()
                print(f"   Sample {i+1}: {state_type} ({'with path' if state_path else 'scratch'})")
        
        # Test performance updates
        print("🧪 Testing performance updates...")
        for checkpoint in saved_checkpoints:
            # Simulate some successful and failed runs
            manager.update_checkpoint_performance(checkpoint.state_path, True, 1.5)
            manager.update_checkpoint_performance(checkpoint.state_path, False, 0.2)
            manager.update_checkpoint_performance(checkpoint.state_path, True, 2.0)
            print(f"   Updated performance for {checkpoint.category}: success_rate={checkpoint.success_rate:.3f}")
        
        # Test curriculum stage update
        print("🧪 Testing curriculum stage updates...")
        manager.update_curriculum_stage(3)
        print(f"✅ Updated to stage 3, current distribution: {manager.current_distribution}")
        
        # Test statistics
        stats = manager.get_statistics()
        print("📊 State Manager Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("✅ All tests passed!")
        return True


def test_integration_config():
    """Test that the configuration integration works."""
    print("\n🧪 Testing configuration integration...")
    
    # Load the updated PPO settings
    config_path = "configs/default_configs/ppo_settings.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Loaded config from {config_path}")
        
        # Check that state curriculum settings are present
        expected_keys = [
            "use_state_curriculum",
            "state_buffer_size", 
            "state_save_directory",
            "catastrophic_forgetting_threshold",
            "validation_frequency"
        ]
        
        for key in expected_keys:
            if key in config:
                print(f"   ✅ {key}: {config[key]}")
            else:
                print(f"   ❌ Missing key: {key}")
                return False
        
        print("✅ Configuration integration test passed!")
        return True
    else:
        print(f"❌ Config file not found: {config_path}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing State Curriculum Implementation\n")
    
    try:
        # Test state manager
        if not test_state_manager():
            print("❌ State manager tests failed")
            return False
        
        # Test configuration
        if not test_integration_config():
            print("❌ Configuration tests failed")
            return False
        
        print("\n🎉 All tests passed! State curriculum system is ready for use.")
        print("\n📋 To enable state curriculum learning:")
        print("   1. Set 'use_state_curriculum': true in ppo_settings.json")
        print("   2. Adjust 'state_buffer_size' as needed (default: 100)")
        print("   3. Set 'state_save_directory' to desired checkpoint location")
        print("   4. Run training normally - the system will automatically manage state sampling")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)