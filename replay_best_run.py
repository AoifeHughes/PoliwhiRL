#!/usr/bin/env python3
"""
Replay Best PPO Run Script

This script takes a database path from a PoliwhiRL PPO training run,
finds the episode with the highest sum of rewards, and replays it
while saving output images to a folder.

Usage:
    python replay_best_run.py /path/to/Database/memory.db [--output-dir results]
"""

import argparse
import sqlite3
import numpy as np
import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import load_default_config, load_user_config, merge_configs
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from PoliwhiRL.agents.PPO.ppo_agent import PPOAgent
from PoliwhiRL.replay.ppo_storage import PPOMemory


class BestRunReplayer:
    def __init__(self, db_path: str, output_dir: str = "replay_output"):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
    
    def find_best_episode(self) -> Tuple[int, float, Dict]:
        """Find the episode with the highest sum of rewards."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if the memory table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory'")
        if not cursor.fetchone():
            conn.close()
            raise ValueError("No 'memory' table found in database. This may not be a PPO database.")
        
        # Get all episodes with their reward data
        cursor.execute("SELECT id, rewards, episode_length FROM memory")
        episodes = cursor.fetchall()
        conn.close()
        
        if not episodes:
            raise ValueError("No episodes found in database")
        
        print(f"Found {len(episodes)} episodes in database")
        
        best_episode_id = None
        best_total_reward = float('-inf')
        episode_stats = []
        
        for episode_id, rewards_blob, episode_length in episodes:
            # Decompress and calculate total reward
            rewards = PPOMemory.decompress_data(rewards_blob, np.float32, (episode_length,))
            total_reward = np.sum(rewards)
            
            episode_stats.append({
                'id': episode_id,
                'total_reward': total_reward,
                'length': episode_length,
                'avg_reward': total_reward / episode_length if episode_length > 0 else 0
            })
            
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_episode_id = episode_id
        
        # Sort episodes by total reward for summary
        episode_stats.sort(key=lambda x: x['total_reward'], reverse=True)
        
        print(f"\nEpisode Performance Summary:")
        print(f"{'Rank':<6} {'Episode ID':<12} {'Total Reward':<15} {'Length':<8} {'Avg Reward':<12}")
        print("-" * 60)
        for i, stats in enumerate(episode_stats[:10]):  # Show top 10
            print(f"{i+1:<6} {stats['id']:<12} {stats['total_reward']:<15.2f} "
                  f"{stats['length']:<8} {stats['avg_reward']:<12.4f}")
        
        if len(episode_stats) > 10:
            print(f"... and {len(episode_stats) - 10} more episodes")
        
        best_stats = episode_stats[0]
        print(f"\nBest Episode: ID {best_episode_id}")
        print(f"Total Reward: {best_total_reward:.2f}")
        print(f"Episode Length: {best_stats['length']} steps")
        print(f"Average Reward per Step: {best_stats['avg_reward']:.4f}")
        
        return best_episode_id, best_total_reward, best_stats
    
    def load_episode_data(self, episode_id: int) -> Dict:
        """Load complete episode data from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memory WHERE id = ?", (episode_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            raise ValueError(f"Episode {episode_id} not found in database")
        
        # Parse the database row
        episode_data = {
            'id': row[0],
            'states_blob': row[1],
            'actions_blob': row[2], 
            'rewards_blob': row[3],
            'dones_blob': row[4],
            'log_probs_blob': row[5],
            'values_blob': row[6],
            'exploration_tensors_blob': row[7],
            'last_next_state_blob': row[8],
            'episode_length': row[9],
            'input_shape': tuple(json.loads(row[10])),
            'sequence_length': row[11]
        }
        
        return episode_data
    
    def decompress_episode_data(self, episode_data: Dict) -> Dict:
        """Decompress episode data for replay."""
        episode_length = episode_data['episode_length']
        input_shape = episode_data['input_shape']
        
        # Decompress all the data
        states = PPOMemory.decompress_data(
            episode_data['states_blob'], np.uint8, 
            (episode_length,) + input_shape
        )
        actions = PPOMemory.decompress_data(
            episode_data['actions_blob'], np.uint8, (episode_length,)
        )
        rewards = PPOMemory.decompress_data(
            episode_data['rewards_blob'], np.float32, (episode_length,)
        )
        dones = PPOMemory.decompress_data(
            episode_data['dones_blob'], np.bool_, (episode_length,)
        )
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'episode_length': episode_length,
            'input_shape': input_shape
        }
    
    def find_config_and_checkpoint(self) -> Tuple[Optional[Dict], Optional[str]]:
        """Try to find associated config and checkpoint files."""
        # Look for config and checkpoint in parent directories
        db_parent = self.db_path.parent
        stage_dir = None
        
        # Try to find the stage directory (should contain Database folder)
        current_dir = db_parent
        for _ in range(3):  # Search up to 3 levels up
            if current_dir.name.startswith('stage_'):
                stage_dir = current_dir
                break
            current_dir = current_dir.parent
            if current_dir == current_dir.parent:  # Reached root
                break
        
        config = None
        checkpoint_dir = None
        stage_number = None
        
        if stage_dir:
            print(f"Found stage directory: {stage_dir}")
            
            # Extract stage number
            try:
                stage_number = int(stage_dir.name.split('_')[1])
                print(f"Detected stage number: {stage_number}")
            except (IndexError, ValueError):
                print("Could not extract stage number from directory name")
            
            # Look for checkpoint
            checkpoint_best = stage_dir / "Checkpoints_best"
            if checkpoint_best.exists():
                checkpoint_dir = str(checkpoint_best)
                print(f"Found best checkpoint: {checkpoint_dir}")
            
            # Try to load config from the stage directory or project root
            config_locations = [
                stage_dir / "config.json",
                stage_dir.parent / "configs" / "default_configs",
                Path(".") / "configs" / "default_configs"
            ]
            
            for config_path in config_locations:
                if config_path.exists():
                    try:
                        if config_path.is_dir():
                            # Load default configs
                            config = self.load_default_configs(config_path)
                        else:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                        print(f"Loaded config from: {config_path}")
                        break
                    except Exception as e:
                        print(f"Failed to load config from {config_path}: {e}")
        
        # Extract original training parameters from checkpoint if available
        if checkpoint_dir and config:
            config = self.apply_original_training_params(config, checkpoint_dir, stage_number)
        
        return config, checkpoint_dir
    
    def load_default_configs(self, config_dir: Path) -> Dict:
        """Load default PoliwhiRL configs."""
        try:
            # Use the main module's function
            return load_default_config()
        except Exception as e:
            print(f"Failed to load default config: {e}")
            # Fallback to manual loading
            config = {}
            config_files = [
                'core_settings.json',
                'ppo_settings.json', 
                'episode_settings.json',
                'reward_settings.json',
                'rom_settings.json',
                'outputs_settings.json'
            ]
            
            for config_file in config_files:
                config_path = config_dir / config_file
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config.update(json.load(f))
            
            return config
    
    def apply_original_training_params(self, config: Dict, checkpoint_dir: str, stage_number: Optional[int]) -> Dict:
        """Extract and apply original training parameters from checkpoint and curriculum."""
        try:
            # Load checkpoint info
            info_path = Path(checkpoint_dir) / "info.pth"
            if info_path.exists():
                import torch
                info = torch.load(info_path, map_location='cpu', weights_only=False)
                
                # Extract curriculum stage info
                saved_goals = info.get('current_n_goals', stage_number or 1)
                print(f"Checkpoint shows {saved_goals} goals target")
                
                # Apply curriculum-based episode lengths (from run_curriculum_single.sh)
                episode_lengths = [50, 50, 500, 600, 4000, 5000, 10000, 12000]
                if stage_number and 1 <= stage_number <= len(episode_lengths):
                    original_episode_length = episode_lengths[stage_number - 1]
                    print(f"Using curriculum episode length for stage {stage_number}: {original_episode_length}")
                    config['episode_length'] = original_episode_length
                
                # Apply stage-specific parameters
                config['N_goals_target'] = saved_goals
                config['break_on_goal'] = True  # Curriculum uses this
                config['punish_steps'] = True   # Curriculum uses this
                
                print(f"Applied original training params:")
                print(f"  - N_goals_target: {config['N_goals_target']}")
                print(f"  - episode_length: {config['episode_length']}")
                print(f"  - break_on_goal: {config['break_on_goal']}")
                print(f"  - punish_steps: {config['punish_steps']}")
                
        except Exception as e:
            print(f"Warning: Could not extract original training params: {e}")
            print("Using default config values")
        
        return config
    
    def create_minimal_config(self, input_shape: Tuple) -> Dict:
        """Create a minimal config for replay when none is found."""
        return {
            'model': 'PPO',
            'device': 'cpu',
            'vision': True,
            'render': True,
            'input_shape': input_shape,
            'sequence_length': 32,
            'episode_length': 10000,
            'ppo_exploration_history_length': 5,
            'N_goals_target': 1,
            'rom_path': '/Users/aoife/git/PoliwhiRL/roms/pokemon_crystal.gb',
            'output_base_dir': str(self.output_dir),
            'record': False
        }
    
    def replay_episode(self, episode_data: Dict, config: Dict, 
                      checkpoint_dir: Optional[str] = None) -> None:
        """Replay the episode and save images."""
        decompressed_data = self.decompress_episode_data(episode_data)
        
        print(f"\nReplaying episode {episode_data['id']}...")
        print(f"Episode length: {decompressed_data['episode_length']} steps")
        print(f"Input shape: {decompressed_data['input_shape']}")
        print(f"Training config: episode_length={config.get('episode_length')}, break_on_goal={config.get('break_on_goal')}, N_goals_target={config.get('N_goals_target')}")
        
        # Create episode output directory
        episode_dir = self.output_dir / f"episode_{episode_data['id']}"
        episode_dir.mkdir(exist_ok=True)
        
        # Set up config for replay - preserve original training parameters
        config = config.copy()
        config['device'] = 'cpu'  # Use CPU for visualization
        config['render'] = True
        config['vision'] = True
        config['record'] = False
        config['output_base_dir'] = str(episode_dir)
        # Keep original break_on_goal and episode_length from training
        # These were extracted from the checkpoint and curriculum settings
        
        try:
            # Create environment
            env = PyBoyEnvironment(config)
            
            # Get environment specifications
            state_shape = env.output_shape()
            num_actions = env.action_space.n
            
            # If we have a checkpoint, load the trained agent
            agent = None
            if checkpoint_dir:
                try:
                    agent = PPOAgent(state_shape, num_actions, config)
                    agent.load_model(checkpoint_dir)
                    print(f"Loaded trained model from {checkpoint_dir}")
                except Exception as e:
                    print(f"Failed to load checkpoint: {e}")
                    print("Proceeding with action replay only")
            
            # Reset environment
            state = env.reset()
            
            total_reward = 0
            images_saved = 0
            
            print("Starting replay...")
            
            for step in range(decompressed_data['episode_length']):
                action = decompressed_data['actions'][step]
                expected_reward = decompressed_data['rewards'][step]
                
                # Take the action
                next_state, actual_reward, done, info = env.step(action)
                total_reward += actual_reward
                
                # Save screenshot
                try:
                    screen = env.get_screen_image(no_resize=True)
                    if screen is not None:
                        # Handle different tensor formats
                        if isinstance(screen, torch.Tensor):
                            screen = screen.cpu().numpy()
                        
                        # Convert tensor to PIL Image
                        if len(screen.shape) == 3:
                            # Convert CHW to HWC for PIL
                            screen = np.transpose(screen, (1, 2, 0))
                        elif len(screen.shape) == 4:
                            # Remove batch dimension and convert CHW to HWC
                            screen = np.transpose(screen[0], (1, 2, 0))
                        
                        # Ensure proper data type
                        screen = np.clip(screen, 0, 255).astype(np.uint8)
                        
                        img = Image.fromarray(screen)
                        img_path = episode_dir / f"step_{step:04d}_action_{action}_reward_{actual_reward:.3f}.png"
                        img.save(img_path)
                        images_saved += 1
                        
                except Exception as e:
                    if step < 10:  # Only print first few errors to avoid spam
                        print(f"Failed to save image at step {step}: {e}")
                
                # Print progress every 100 steps
                if step % 100 == 0:
                    print(f"Step {step}/{decompressed_data['episode_length']}, "
                          f"Action: {action}, Reward: {actual_reward:.3f}, "
                          f"Total: {total_reward:.3f}")
                
                if done:
                    print(f"Episode terminated early at step {step} (done=True)")
                    break
                
                state = next_state
            
            env.close()
            
            print(f"\nReplay completed!")
            print(f"Total reward achieved: {total_reward:.3f}")
            print(f"Expected total reward: {np.sum(decompressed_data['rewards']):.3f}")
            print(f"Images saved: {images_saved}")
            print(f"Images saved to: {episode_dir}")
            
            # Save episode summary
            summary = {
                'episode_id': int(episode_data['id']),
                'episode_length': int(decompressed_data['episode_length']),
                'total_reward_achieved': float(total_reward),
                'total_reward_expected': float(np.sum(decompressed_data['rewards'])),
                'images_saved': int(images_saved),
                'output_directory': str(episode_dir),
                'config_used': {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item()) 
                               for k, v in config.items()}
            }
            
            summary_path = episode_dir / "replay_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            print(f"Error during replay: {e}")
            raise
    
    def run(self) -> None:
        """Main execution function."""
        print(f"Analyzing database: {self.db_path}")
        
        # Find the best episode
        best_episode_id, best_reward, best_stats = self.find_best_episode()
        
        # Load episode data
        episode_data = self.load_episode_data(best_episode_id)
        
        # Try to find config and checkpoint
        config, checkpoint_dir = self.find_config_and_checkpoint()
        
        if config is None:
            print("Warning: No config found, creating minimal config")
            config = self.create_minimal_config(episode_data['input_shape'])
        
        # Replay the episode
        self.replay_episode(episode_data, config, checkpoint_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Replay the best performing episode from a PoliwhiRL PPO database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python replay_best_run.py stage_1/Database/memory.db
    python replay_best_run.py /path/to/Database/memory.db --output-dir my_replay
        """
    )
    
    parser.add_argument('database_path', type=str,
                       help='Path to the PPO memory database (memory.db)')
    parser.add_argument('--output-dir', type=str, default='replay_output',
                       help='Output directory for replay images and data (default: replay_output)')
    
    args = parser.parse_args()
    
    try:
        replayer = BestRunReplayer(args.database_path, args.output_dir)
        replayer.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()