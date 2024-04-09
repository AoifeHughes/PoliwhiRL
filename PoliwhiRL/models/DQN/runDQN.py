import numpy as np
from tqdm import tqdm

from .DQN import DQNAgent
from PoliwhiRL.environment.controller import Controller as Env
from PoliwhiRL.environment.controller import action_space
from PoliwhiRL.utils import plot_best_attempts

def run_model(config):
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
    checkpoint = config.get("checkpoint", "final_model.pth")

    # Create the DQN agent
    agent = DQNAgent(state_size, action_size, batch_size, gamma, lr, epsilon, epsilon_decay, epsilon_min, memory_size, device)
    try:
        agent.load(checkpoint)
        print(f"Loaded model from {checkpoint}")
    except FileNotFoundError as e:
        print(e)
        print("No model found, training from scratch")

    rewards = []
    best_reward = float("-inf")

    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        done = False
        episode_transitions = []

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            if episode % 25== 0:
                env.record(epsilon, "dqn", 0, reward)
            episode_transitions.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        if episode_reward > best_reward:
            best_reward = episode_reward

        # Add the entire episode to memory
        for transition in episode_transitions:
            agent.memorize(*transition)

        if len(agent.memory) >= batch_size:
            agent.replay()

        tqdm.write(f"Episode: {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Best Reward: {best_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        if (episode + 1) % 100 == 0:
            agent.save(f"model_checkpoint_{episode+1}.pth")
            plot_best_attempts("./results/", "DQN", 0, rewards)

    # Save the final trained model
    agent.save(checkpoint)

    # Close the environment
    env.close()