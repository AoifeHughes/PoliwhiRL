# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from .DQN import DQNAgent
from PoliwhiRL.environment.controller import Controller as Env
from PoliwhiRL.environment.controller import action_space
from PoliwhiRL.utils import plot_best_attempts


def run_curriculum(config):
    done_lims = [1, 3, 5]
    for i in range(3):
        print(f"Running curriculum {i}")
        mutli = 10 * i if i > 0 else 1
        config["episode_length"] = config["episode_length"] * mutli
        run_model(config, i, done_lim=done_lims[i])


def run_model(config, record_id=0, done_lim=1000):
    env = Env(config)
    # Define the hyperparameters
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

    # Create the DQN agent
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
        db_location
    )

    try:
        agent.load("final_model.pth")
        print("Loaded model from final_model.pth")
    except FileNotFoundError as e:
        print(e)
        print("No model found, training from scratch")

    rewards = []
    episode_lengths = []
    best_reward = float("-inf")

    # Training loop
    for episode in tqdm(range(num_episodes+random_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        state_sequence = []
        action_sequence = []
        reward_sequence = []
        next_state_sequence = []
        done_sequence = []

        while not done:
            state_sequence.append(state)
            action = agent.act(np.array(state_sequence))
            next_state, reward, done = env.step(action)
            if np.sum(reward_sequence) >= done_lim:
                done = True
            env.record(epsilon, f"dqn{record_id}", 0, reward)
            action_sequence.append(action)
            reward_sequence.append(reward)
            next_state_sequence.append(next_state)
            done_sequence.append(done)
            state = next_state
            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_reward > best_reward:
            best_reward = episode_reward

        # Add the entire episode to memory
        for i in range(len(action_sequence)):
            agent.memorize(
                state_sequence[i],
                action_sequence[i],
                reward_sequence[i],
                next_state_sequence[i],
                done_sequence[i],
            )

        if len(agent.memory) >= batch_size and episode > random_episodes:
            agent.replay()

        avg_reward = np.mean(rewards[-100:])

        tqdm.write(
            f"Episode: {episode+1}/{num_episodes+random_episodes}, Reward: {episode_reward:.2f}, Best Reward: {best_reward:.2f}, "
            f"Avg Reward (100 eps): {avg_reward:.2f},  Epsilon: {agent.epsilon:.2f}"
        )
        if (episode + 1) % 50 == 0:
            plot_best_attempts("./results/", f"DQN{record_id}", 0, rewards)

    # Save the final trained model
    agent.save("final_model.pth")
    # Close the environment
    env.close()
