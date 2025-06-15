#!/usr/bin/env python3
"""
Analyze and visualize the best performing agents from each curriculum stage.
This script finds the best checkpoint from each stage and re-runs it with visual output.
"""

import json
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from PIL import Image

# Add the project root to the path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import parse_args, load_config
from PoliwhiRL.environment.gym_env import PokemonEnvironment
from PoliwhiRL.agents.PPO.ppo_agent import PPOAgent
from PoliwhiRL.utils.visuals import plot_metrics


class BestPerformerAnalyzer:
    def __init__(self, output_dir: str = "best_performer_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.stage_data = {}
        
    def find_stage_directories(self) -> List[str]:
        """Find all stage directories in the current directory."""
        stage_dirs = []
        for i in range(1, 9):  # Check stages 1-8
            if os.path.exists(f"stage_{i}") and os.path.exists(f"stage_{i}/Checkpoints_best"):
                stage_dirs.append(f"stage_{i}")
        return sorted(stage_dirs)
    
    def load_stage_performance(self, stage_dir: str) -> Dict:
        """Load performance data for a given stage."""
        stage_num = int(stage_dir.split('_')[1])
        
        # Load failure learning stats
        stats_path = os.path.join(stage_dir, "Results", "failure_learning_stats.json")
        if not os.path.exists(stats_path):
            print(f"Warning: No stats file found for {stage_dir}")
            return None
            
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Load checkpoint info
        info_path = os.path.join(stage_dir, "Checkpoints_best", "info.pth")
        if os.path.exists(info_path):
            info = torch.load(info_path, map_location='cpu')
        else:
            info = {}
        
        # Extract raw performance data
        raw_performance = stats.get('failure_learning_data', {}).get('research_stats', {}).get('raw_performance', [])
        
        # Find best episode
        if raw_performance:
            best_reward = max(raw_performance)
            best_episode = raw_performance.index(best_reward)
        else:
            best_reward = info.get('best_reward', 0)
            best_episode = info.get('best_episode', 0)
        
        return {
            'stage': stage_num,
            'stage_dir': stage_dir,
            'total_episodes': stats.get('session_summary', {}).get('total_episodes', 0),
            'best_reward': best_reward,
            'best_episode': best_episode,
            'raw_performance': raw_performance,
            'info': info
        }
    
    def plot_performance_comparison(self, stage_data_list: List[Dict]):
        """Create a comparison plot of performance across stages."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Curriculum Learning Performance Analysis', fontsize=16)
        
        # Plot 1: Best reward per stage
        ax1 = axes[0, 0]
        stages = [d['stage'] for d in stage_data_list]
        best_rewards = [d['best_reward'] for d in stage_data_list]
        ax1.bar(stages, best_rewards, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Stage (Number of Goals)')
        ax1.set_ylabel('Best Reward Achieved')
        ax1.set_title('Peak Performance by Stage')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Best episode number per stage
        ax2 = axes[0, 1]
        best_episodes = [d['best_episode'] for d in stage_data_list]
        ax2.plot(stages, best_episodes, 'o-', color='green', markersize=8)
        ax2.set_xlabel('Stage (Number of Goals)')
        ax2.set_ylabel('Episode Number of Best Performance')
        ax2.set_title('When Best Performance Occurred')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning curves overlay
        ax3 = axes[1, 0]
        for data in stage_data_list:
            if data['raw_performance']:
                # Smooth the curve with rolling average
                performance = np.array(data['raw_performance'])
                window = min(20, len(performance) // 10)
                if window > 1:
                    smoothed = np.convolve(performance, np.ones(window)/window, mode='valid')
                    episodes = np.arange(len(smoothed))
                    ax3.plot(episodes, smoothed, label=f"Stage {data['stage']}", alpha=0.7)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.set_title('Learning Curves (Smoothed)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        for data in stage_data_list:
            table_data.append([
                f"Stage {data['stage']}",
                f"{data['best_reward']:.2f}",
                f"{data['best_episode']}",
                f"{data['total_episodes']}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Stage', 'Best Reward', 'Best Episode', 'Total Episodes'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        save_path = self.output_dir / 'performance_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved performance comparison to {save_path}")
    
    def run_best_checkpoint(self, stage_data: Dict, episodes: int = 1, 
                          save_video: bool = False, save_screenshots: bool = True):
        """Re-run the best checkpoint from a stage with visual output."""
        stage_num = stage_data['stage']
        checkpoint_dir = os.path.join(stage_data['stage_dir'], 'Checkpoints_best')
        
        if not os.path.exists(checkpoint_dir):
            print(f"No best checkpoint found for stage {stage_num}")
            return
        
        print(f"\nRunning best checkpoint for Stage {stage_num} (Episode {stage_data['best_episode']})")
        print(f"Best reward: {stage_data['best_reward']:.2f}")
        
        # Create output directory for this stage
        stage_output_dir = self.output_dir / f"stage_{stage_num}_visualization"
        stage_output_dir.mkdir(exist_ok=True)
        
        # Prepare arguments for running the agent
        args = [
            '--model', 'PPO',
            '--device', 'cpu',  # Use CPU for visualization
            '--vision', 'true',
            '--num_episodes', str(episodes),
            '--N_goals_target', str(stage_num),
            '--load_checkpoint', checkpoint_dir,
            '--output_base_dir', str(stage_output_dir),
            '--ppo_num_agents', '1',
            '--break_on_goal', 'true',
            '--record', 'true' if save_video else 'false'
        ]
        
        # Parse args and load config
        parsed_args = parse_args(args)
        config = load_config(parsed_args)
        
        # Override some settings for visualization
        config['vision'] = True
        config['render'] = True
        config['record'] = save_video
        config['output_base_dir'] = str(stage_output_dir)
        config['episode_length'] = 10000  # Allow longer episodes for visualization
        
        # Create environment and agent
        env = PokemonEnvironment(config)
        agent = PPOAgent(env, config)
        
        # Load the best checkpoint
        agent.load_model(checkpoint_dir)
        
        # Enable recording if requested
        if save_video:
            env.enable_record(str(stage_output_dir))
        
        # Enable rendering for screenshots
        if save_screenshots:
            env.enable_render()
        
        # Run evaluation episodes
        print(f"Starting visual evaluation for stage {stage_num}...")
        
        episode_results = []
        for episode in range(episodes):
            # Set recording path for this episode
            record_path = str(stage_output_dir / f"episode_{episode}") if save_video else None
            
            # Run the episode using the agent's method
            episode_data = agent.run_episode(record_loc=record_path)
            
            # Extract results
            total_reward = episode_data.get('total_reward', 0)
            steps = episode_data.get('episode_length', 0)
            goals_reached = episode_data.get('goals_reached', 0)
            
            episode_results.append({
                'episode': episode,
                'reward': total_reward,
                'steps': steps,
                'goals_reached': goals_reached
            })
            
            print(f"  Episode {episode}: Reward = {total_reward:.2f}, Steps = {steps}, Goals = {goals_reached}")
            
            # Save key screenshots during the episode if requested
            if save_screenshots:
                try:
                    # Save final state screenshot
                    screen = env.get_screen_image(no_resize=True)
                    if screen is not None:
                        if len(screen.shape) == 3:
                            # Convert CHW to HWC for PIL
                            screen = np.transpose(screen, (1, 2, 0))
                        img = Image.fromarray(screen.astype(np.uint8))
                        screenshot_path = stage_output_dir / f"episode_{episode}_final.png"
                        img.save(screenshot_path)
                except Exception as e:
                    print(f"    Warning: Could not save screenshot: {e}")
        
        env.close()
        
        # Save episode results
        results_file = stage_output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'stage': stage_num,
                'best_episode_original': stage_data['best_episode'],
                'best_reward_original': stage_data['best_reward'],
                'evaluation_episodes': episode_results,
                'config_used': {
                    'episode_length': config['episode_length'],
                    'N_goals_target': config['N_goals_target'],
                    'device': config['device']
                }
            }, f, indent=2)
        
        return stage_output_dir
    
    def generate_summary_report(self, stage_data_list: List[Dict]):
        """Generate a markdown summary report."""
        report_path = self.output_dir / 'performance_summary.md'
        
        with open(report_path, 'w') as f:
            f.write("# Curriculum Learning Performance Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"- Total stages analyzed: {len(stage_data_list)}\n")
            f.write(f"- Stages found: {', '.join([str(d['stage']) for d in stage_data_list])}\n\n")
            
            f.write("## Best Performance by Stage\n\n")
            f.write("| Stage | Goals | Best Reward | Best Episode | Total Episodes | Improvement |\n")
            f.write("|-------|-------|-------------|--------------|----------------|-------------|\n")
            
            prev_reward = 0
            for i, data in enumerate(stage_data_list):
                improvement = f"{((data['best_reward'] - prev_reward) / (prev_reward if prev_reward > 0 else 1)) * 100:.1f}%" if i > 0 else "N/A"
                f.write(f"| {data['stage']} | {data['stage']} | "
                       f"{data['best_reward']:.2f} | {data['best_episode']} | "
                       f"{data['total_episodes']} | {improvement} |\n")
                prev_reward = data['best_reward']
            
            f.write("\n## Analysis\n\n")
            
            # Find most efficient stage (best reward per episode)
            efficiency_scores = [(d['best_reward'] / (d['best_episode'] + 1), d['stage']) 
                               for d in stage_data_list if d['best_episode'] > 0]
            if efficiency_scores:
                best_efficiency, best_stage = max(efficiency_scores)
                f.write(f"- Most efficient learning: Stage {best_stage} "
                       f"(efficiency score: {best_efficiency:.4f})\n")
            
            # Find stage with highest reward
            best_overall = max(stage_data_list, key=lambda x: x['best_reward'])
            f.write(f"- Highest reward achieved: Stage {best_overall['stage']} "
                   f"with reward {best_overall['best_reward']:.2f}\n")
            
            # Learning speed analysis
            early_learners = [d for d in stage_data_list if d['best_episode'] < 100]
            if early_learners:
                f.write(f"- Fast learners (best within 100 episodes): "
                       f"Stages {', '.join([str(d['stage']) for d in early_learners])}\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the analysis:\n\n")
            
            # Check if performance plateaus
            if len(stage_data_list) >= 3:
                recent_rewards = [d['best_reward'] for d in stage_data_list[-3:]]
                if max(recent_rewards) - min(recent_rewards) < 0.1 * max(recent_rewards):
                    f.write("- Performance appears to be plateauing in recent stages. "
                           "Consider adjusting hyperparameters or reward structure.\n")
                else:
                    f.write("- Good progression observed across stages.\n")
            
            # Check learning efficiency
            avg_best_episode = np.mean([d['best_episode'] for d in stage_data_list])
            if avg_best_episode < 200:
                f.write("- Agent learns quickly on average. Current hyperparameters seem effective.\n")
            else:
                f.write("- Agent takes many episodes to reach peak performance. "
                       "Consider increasing learning rate or exploration.\n")
        
        print(f"Saved summary report to {report_path}")
    
    def analyze_all_stages(self, run_visualization: bool = True, 
                          episodes_per_stage: int = 1):
        """Main analysis function."""
        print("Finding curriculum stages...")
        stage_dirs = self.find_stage_directories()
        
        if not stage_dirs:
            print("No stage directories found!")
            return
        
        print(f"Found {len(stage_dirs)} stages: {', '.join(stage_dirs)}")
        
        # Load performance data for each stage
        stage_data_list = []
        for stage_dir in stage_dirs:
            data = self.load_stage_performance(stage_dir)
            if data:
                stage_data_list.append(data)
                self.stage_data[data['stage']] = data
        
        if not stage_data_list:
            print("No performance data found!")
            return
        
        # Generate comparison plots
        print("\nGenerating performance comparison plots...")
        self.plot_performance_comparison(stage_data_list)
        
        # Generate summary report
        print("Generating summary report...")
        self.generate_summary_report(stage_data_list)
        
        # Run best checkpoints with visualization
        if run_visualization:
            print("\nRunning best checkpoints with visualization...")
            for data in stage_data_list:
                try:
                    self.run_best_checkpoint(data, episodes=episodes_per_stage, 
                                           save_screenshots=True, save_video=False)
                except Exception as e:
                    print(f"Error running stage {data['stage']}: {e}")
        
        print(f"\nAnalysis complete! Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze best performers from curriculum learning')
    parser.add_argument('--output-dir', type=str, default='best_performer_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip running visual evaluation of best checkpoints')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to run per stage for visualization')
    parser.add_argument('--stages', type=str, default=None,
                       help='Comma-separated list of stages to analyze (e.g., "1,3,5")')
    
    args = parser.parse_args()
    
    analyzer = BestPerformerAnalyzer(output_dir=args.output_dir)
    
    # Run analysis
    analyzer.analyze_all_stages(
        run_visualization=not args.no_visualization,
        episodes_per_stage=args.episodes
    )


if __name__ == "__main__":
    main()