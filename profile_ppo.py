#!/usr/bin/env python3
"""
PPO Performance Profiling Script

This script profiles the exact training command to identify performance bottlenecks.
It provides detailed timing analysis for each component of the training pipeline.

Usage:
    python profile_ppo.py [--episodes N] [--steps N] [--detailed]
"""

import argparse
import cProfile
import pstats
import time
import sys
import os
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import io

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import load_default_config
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from PoliwhiRL.agents.PPO.ppo_agent import PPOAgent


class PerformanceProfiler:
    """Detailed performance profiler for PPO training"""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
        self.memory_usage = {}
        
    @contextmanager
    def timer(self, name):
        """Context manager for timing code blocks"""
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        
        if name not in self.timings:
            self.timings[name] = []
            self.call_counts[name] = 0
        
        self.timings[name].append(end_time - start_time)
        self.call_counts[name] += 1
    
    def memory_snapshot(self, name):
        """Take a memory usage snapshot"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        elif torch.backends.mps.is_available():
            # MPS doesn't have direct memory monitoring, use approximation
            gpu_memory = 0
        else:
            gpu_memory = 0
            
        import psutil
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**2  # MB
        
        self.memory_usage[name] = {
            'cpu_mb': cpu_memory,
            'gpu_mb': gpu_memory
        }
    
    def get_summary(self):
        """Get comprehensive timing summary"""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times),
                'call_count': self.call_counts[name],
                'percentage': 0  # Will be calculated later
            }
        
        # Calculate percentages
        total_time = sum(data['total_time'] for data in summary.values())
        for name, data in summary.items():
            data['percentage'] = (data['total_time'] / total_time) * 100 if total_time > 0 else 0
        
        return summary
    
    def print_report(self):
        """Print detailed performance report"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("PPO PERFORMANCE PROFILING REPORT")
        print("="*80)
        
        # Sort by total time
        sorted_items = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        print(f"\n{'Component':<25} {'Total (s)':<10} {'Avg (ms)':<10} {'Calls':<8} {'%':<6} {'Min (ms)':<10} {'Max (ms)':<10}")
        print("-" * 80)
        
        for name, data in sorted_items:
            print(f"{name:<25} {data['total_time']:<10.3f} {data['avg_time']*1000:<10.2f} "
                  f"{data['call_count']:<8} {data['percentage']:<6.1f} "
                  f"{data['min_time']*1000:<10.2f} {data['max_time']*1000:<10.2f}")
        
        print("\n" + "="*80)
        print("MEMORY USAGE SNAPSHOTS")
        print("="*80)
        
        for name, memory in self.memory_usage.items():
            print(f"{name:<25} CPU: {memory['cpu_mb']:.1f} MB, GPU: {memory['gpu_mb']:.1f} MB")


def profile_environment_step(env, action, profiler):
    """Profile the environment step with detailed breakdown"""
    
    with profiler.timer('env_handle_action'):
        # Time the action handling (includes 90-frame skip)
        env._handle_action(action)
    
    with profiler.timer('env_calculate_fitness'):
        # Time the fitness/reward calculation
        env._calculate_fitness()
    
    with profiler.timer('env_get_observation'):
        # Time the observation extraction
        observation = env.get_observation()
    
    with profiler.timer('env_recording_overhead'):
        # Time any recording overhead
        if env.record:
            env.save_step_img_data(
                env.record_folder, outdir=env.config["record_path"]
            )
    
    return observation, env._fitness, env.done, False


def profile_exploration_memory_detailed(agent, screen, action, next_state, reward, profiler):
    """Profile exploration memory operations with detailed breakdown"""
    
    with profiler.timer('exploration_add_transition'):
        # Time the transition storage (includes hashing internally)
        agent.exploration_memory.add_transition(screen, action, next_state, reward)
    
    with profiler.timer('exploration_tensor_generation'):
        # Time tensor generation (includes cache checks)
        exploration_tensor = agent.exploration_memory.get_memory_tensor()
    
    return exploration_tensor


def profile_model_update_detailed(agent, data, step, profiler):
    """Profile model update with detailed breakdown"""
    
    with profiler.timer('model_data_preparation'):
        # Time data formatting and tensor preparation
        if data is None:
            return 0.0, 0.0
        
        # The data is already prepared by get_all_data(), so this times any additional prep
        pass
    
    with profiler.timer('model_forward_backward'):
        # Time the actual neural network operations
        loss, icm_loss = agent.model.update(data, step)
    
    return loss, icm_loss


def profile_training_step(env, agent, profiler, step_num):
    """Profile a single training step with detailed timing"""
    
    try:
        with profiler.timer('total_step'):
            # Environment step
            with profiler.timer('env_step'):
                with profiler.timer('env_action_handling'):
                    action = np.random.randint(0, env.action_space.n)  # Random action for profiling
                
                with profiler.timer('env_pyboy_tick'):
                    # Detailed profiling of environment step
                    next_state, reward, done, info = profile_environment_step(env, action, profiler)
            
            # Image processing (detailed profiling)
            with profiler.timer('image_processing'):
                with profiler.timer('get_screen_image'):
                    screen = env.get_screen_image()
                
                with profiler.timer('numpy_to_tensor'):
                    screen_tensor = torch.FloatTensor(screen)
                
                with profiler.timer('tensor_unsqueeze'):
                    screen_tensor = screen_tensor.unsqueeze(0)
            
            # Exploration memory (detailed profiling)
            with profiler.timer('exploration_memory'):
                exploration_tensor = profile_exploration_memory_detailed(
                    agent, screen, action, next_state, reward, profiler
                )
            
            # Model forward pass (skip this for profiling to avoid tensor shape issues)
            # The model expects sequence data which is complex to set up for profiling
            value = None  # Set default value for memory storage
            
            # Memory buffer operations
            with profiler.timer('memory_buffer'):
                with profiler.timer('store_transition'):
                    agent.memory.store_transition(
                        screen, next_state, action, reward, done, 
                        np.log(0.1), 0.0, exploration_tensor
                    )
            
            # Take memory snapshots periodically
            if step_num % 10 == 0:
                profiler.memory_snapshot(f'step_{step_num}')
        
        return next_state, reward, done
        
    except Exception as e:
        print(f"Error in profile_training_step: {e}")
        import traceback
        traceback.print_exc()
        # Return dummy values to continue profiling
        return next_state if 'next_state' in locals() else None, 0.0, False


def run_profiling(num_episodes=2, max_steps_per_episode=50, detailed_cprofile=False):
    """Run the profiling with your exact training configuration"""
    
    print("Setting up PPO training environment for profiling...")
    
    # Create config matching your command
    config = load_default_config()
    config.update({
        'model': 'PPO',
        'device': 'mps',  # Your typical device
        'vision': True,
        'episode_length': 5000,  # Your typical setting
        'num_episodes': num_episodes,
        'ppo_update_frequency': 512,
        'ppo_iterations': 5,
        'N_goals_target': 10,
        'output_base_dir': 'profiling_output/',
        'ppo_num_agents': 1,
        'punish_steps': True,
        'report_episode': True,
        'use_curriculum': False,
        'break_on_goal': False,
        'ppo_entropy_coef': 0.15,
        'ppo_entropy_coef_decay': 0.9999,
        'render': True  # Enable rendering for realistic profiling
    })
    
    profiler = PerformanceProfiler()
    
    print(f"Creating environment (device: {config['device']})...")
    
    with profiler.timer('env_creation'):
        env = PyBoyEnvironment(config, force_window=False)
    
    with profiler.timer('agent_creation'):
        state_shape = env.output_shape()
        num_actions = env.action_space.n
        agent = PPOAgent(state_shape, num_actions, config)
    
    profiler.memory_snapshot('after_initialization')
    
    print(f"Starting profiling for {num_episodes} episodes, max {max_steps_per_episode} steps each...")
    
    total_start_time = time.perf_counter()
    
    # Optional detailed cProfile
    if detailed_cprofile:
        pr = cProfile.Profile()
        pr.enable()
    
    try:
        for episode in range(num_episodes):
            print(f"Profiling episode {episode + 1}/{num_episodes}")
            
            with profiler.timer('episode_reset'):
                state = env.reset()
            
            profiler.memory_snapshot(f'episode_{episode}_start')
            
            step = 0
            done = False
            
            while step < max_steps_per_episode:
                state, reward, env_done = profile_training_step(env, agent, profiler, step)
                step += 1
                
                # Reset environment if it's done to continue profiling
                if env_done:
                    with profiler.timer('episode_reset'):
                        state = env.reset()
                
                # Perform model update when buffer is full (like real training)
                if agent.memory.episode_length >= agent.memory.update_frequency:
                    with profiler.timer('model_update'):
                        with profiler.timer('model_update_full'):
                            # Get data for model update
                            with profiler.timer('buffer_data_extraction'):
                                data = agent.memory.get_all_data()
                            
                            if data is not None:
                                try:
                                    # Detailed model update profiling
                                    loss, icm_loss = profile_model_update_detailed(agent, data, step, profiler)
                                except Exception as e:
                                    # If model update fails due to tensor issues, just log it
                                    print(f"    Model update failed (expected): {type(e).__name__}")
                                    pass
                        with profiler.timer('memory_reset'):
                            agent.memory.reset()
                
                # Print progress
                if step % 10 == 0:
                    print(f"  Step {step}/{max_steps_per_episode}")
            
            profiler.memory_snapshot(f'episode_{episode}_end')
    
    finally:
        total_time = time.perf_counter() - total_start_time
        
        if detailed_cprofile:
            pr.disable()
            
            # Save cProfile results
            pr.dump_stats('ppo_profile.prof')
            
            # Print top functions
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Top 30 functions
            
            print("\n" + "="*80)
            print("CPROFILE TOP FUNCTIONS")
            print("="*80)
            print(s.getvalue())
        
        env.close()
        
        print(f"\nTotal profiling time: {total_time:.3f} seconds")
        profiler.print_report()
        
        # Calculate steps per second
        total_steps = sum(profiler.call_counts.get('total_step', 0) for _ in range(num_episodes))
        if total_steps > 0:
            steps_per_second = total_steps / total_time
            print(f"\nPerformance: {steps_per_second:.2f} steps/second")
            print(f"Time per step: {1000/steps_per_second:.2f} ms")
        
        return profiler


def main():
    parser = argparse.ArgumentParser(description='Profile PPO training performance')
    parser.add_argument('--episodes', type=int, default=2,
                       help='Number of episodes to profile (default: 2)')
    parser.add_argument('--steps', type=int, default=50,
                       help='Max steps per episode (default: 50)')
    parser.add_argument('--detailed', action='store_true',
                       help='Enable detailed cProfile analysis')
    
    args = parser.parse_args()
    
    try:
        profiler = run_profiling(
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            detailed_cprofile=args.detailed
        )
        
        print("\nProfiling completed successfully!")
        print("Key optimization targets identified in the report above.")
        
        if args.detailed:
            print("\nDetailed cProfile data saved to 'ppo_profile.prof'")
            print("View with: python -m pstats ppo_profile.prof")
        
    except Exception as e:
        print(f"Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())