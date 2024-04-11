# -*- coding: utf-8 -*-
from .parallel_agent import ParallelDQNAgent
from .agent import DQNAgent
from PoliwhiRL.environment.controller import Controller as Env
from PoliwhiRL.environment.controller import action_space

def run_model(config, record_id=0, done_lim=1000):
    env = Env(config)
    state_size = (1 if config.get("use_grayscale", False) else 3, *env.screen_size())
    action_size = len(action_space)
    batch_size = config.get("batch_size", 8)
    gamma = config.get("gamma", 0.99)
    lr = config.get("learning_rate", 0.001)
    epsilon = config.get("epsilon", 1.0)
    epsilon_decay = config.get("epsilon_decay", 0.99)
    epsilon_min = config.get("epsilon_min", 0.01)
    memory_size = config.get("memory_size", 10000)
    num_episodes = config.get("num_episodes", 1000)
    device = config.get("device", "cpu")
    random_episodes = config.get("random_episodes", 0)
    db_location = config.get("database_location", "./database/memory.db")
    num_workers = config.get("num_workers", 1)

    if num_workers > 1:
        config['device'] = 'cpu'  
        agent = ParallelDQNAgent(
            num_workers,
            state_size,
            action_size,
            batch_size,
            gamma,
            lr,
            epsilon,
            epsilon_decay,
            epsilon_min,
            memory_size,
            device,
            db_location,
            config,
        )
    else:
        # Use DQNAgent for single process
        agent = DQNAgent(
            state_size,
            action_size,
            batch_size,
            gamma,
            lr,
            epsilon,
            epsilon_decay,
            epsilon_min,
            memory_size,
            device,
            db_location,
        )

    try:
        agent.load("final_model.pth")
        print("Loaded model from final_model.pth")
    except FileNotFoundError as e:
        print(e)
        print("No model found, training from scratch")

    if num_workers > 1:
        # Train the parallel agent
        agent.train(num_episodes)
    else:
        # Train the single process agent
        agent.train(env, num_episodes, random_episodes, done_lim, record_id)

    # Close the environment
    env.close()
    agent.save("final_model.pth")