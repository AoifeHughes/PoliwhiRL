#!/usr/bin/env python3
"""
Performance Improvement Benchmark

This script compares performance before and after optimizations
by running multiple timing tests on key components.
"""

import time
import numpy as np
import sys
import os
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import load_default_config
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from PoliwhiRL.agents.PPO.ppo_agent import PPOAgent


@contextmanager
def timer():
    """Simple timing context manager"""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    

def benchmark_image_processing(env, num_iterations=100):
    """Benchmark image processing performance"""
    print(f"Benchmarking image processing ({num_iterations} iterations)...")
    
    times = []
    for _ in range(num_iterations):
        with timer() as get_time:
            screen = env.get_screen_image()
        times.append(get_time())
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    
    print(f"  Average time: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"  Range: {min_time:.3f} - {max_time:.3f} ms")
    print(f"  Throughput: {1000/avg_time:.1f} images/second")
    
    return avg_time


def benchmark_exploration_memory(agent, num_iterations=100):
    """Benchmark exploration memory tensor generation"""
    print(f"Benchmarking exploration memory ({num_iterations} iterations)...")
    
    # Pre-populate some memory
    for i in range(20):
        dummy_state = np.random.randint(0, 255, (3, 72, 80), dtype=np.uint8)
        agent.exploration_memory.add_transition(dummy_state, i % 9, dummy_state, 0.1)
    
    times = []
    for _ in range(num_iterations):
        with timer() as get_time:
            tensor = agent.exploration_memory.get_memory_tensor()
        times.append(get_time())
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    print(f"  Average time: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"  Throughput: {1000/avg_time:.1f} tensors/second")
    
    return avg_time


def benchmark_full_step(env, agent, num_iterations=50):
    """Benchmark a complete environment step"""
    print(f"Benchmarking full environment step ({num_iterations} iterations)...")
    
    times = []
    state = env.reset()
    
    for i in range(num_iterations):
        action = i % env.action_space.n
        
        with timer() as get_time:
            # Full step including all processing
            next_state, reward, done, info = env.step(action)
            screen = env.get_screen_image()
            agent.exploration_memory.add_transition(state, action, next_state, reward)
            exploration_tensor = agent.exploration_memory.get_memory_tensor()
        
        times.append(get_time())
        state = next_state
        
        if done:
            state = env.reset()
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    print(f"  Average time: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"  Steps per second: {1000/avg_time:.1f}")
    
    return avg_time


def main():
    print("Performance Optimization Benchmark")
    print("=" * 50)
    
    # Setup
    config = load_default_config()
    config.update({
        'device': 'mps',
        'vision': True,
        'episode_length': 1000,
        'render': True,
        'use_grayscale': False,
        'scaling_factor': 0.5
    })
    
    print("Setting up environment and agent...")
    env = PyBoyEnvironment(config, force_window=False)
    state_shape = env.output_shape()
    num_actions = env.action_space.n
    agent = PPOAgent(state_shape, num_actions, config)
    
    print(f"Configuration:")
    print(f"  Device: {config['device']}")
    print(f"  Vision: {config['vision']}")
    print(f"  Grayscale: {config['use_grayscale']}")
    print(f"  Scaling factor: {config['scaling_factor']}")
    print(f"  State shape: {state_shape}")
    print()
    
    # Run benchmarks
    try:
        img_time = benchmark_image_processing(env, 100)
        print()
        
        mem_time = benchmark_exploration_memory(agent, 100)
        print()
        
        step_time = benchmark_full_step(env, agent, 50)
        print()
        
        # Summary
        print("=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Image processing:      {img_time:.3f} ms")
        print(f"Exploration memory:    {mem_time:.3f} ms")
        print(f"Full step:             {step_time:.3f} ms")
        print()
        print(f"Estimated training speed: {1000/step_time:.1f} steps/second")
        
        # Calculate estimated time for your typical training
        steps_per_episode = 5000  # Your typical episode length
        num_episodes = 100        # Your typical training
        total_steps = steps_per_episode * num_episodes
        estimated_time_hours = (total_steps * step_time / 1000) / 3600
        
        print(f"Estimated time for {num_episodes} episodes of {steps_per_episode} steps:")
        print(f"  {estimated_time_hours:.2f} hours")
        
    finally:
        env.close()


if __name__ == "__main__":
    main()