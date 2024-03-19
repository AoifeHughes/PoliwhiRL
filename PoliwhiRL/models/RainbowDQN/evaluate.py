import torch
from PoliwhiRL.utils import image_to_tensor

def evaluate_model(config, env, policy_net):
    """
    Evaluates the model's performance by running it through a set number of episodes in the environment,
    using the policy network to determine the best action at each step.

    Args:
        config (dict): Configuration parameters, including device and evaluation settings.
        env (Controller): The environment in which to evaluate the model.
        policy_net (RainbowDQN): The trained policy network.

    Returns:
        float: The average reward obtained over all evaluation episodes.
    """
    policy_net.eval()  # Ensure the model is in evaluation mode.
    total_rewards = 0
    num_episodes = config.get("eval_episodes", 10)  # Default to 10 episodes for evaluation if not specified.

    for episode in range(num_episodes):
        state = env.reset()
        state = image_to_tensor(state, config["device"])  # Convert state to tensor
        episode_rewards = 0
        done = False

        while not done:
            with torch.no_grad():  # No need to track gradients
                q_values = policy_net(state.unsqueeze(0).unsqueeze(0).to(config["device"]))
                action = torch.argmax(q_values, dim=1).item()  # Select the action with the highest Q-value

            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, config["device"])  # Convert next state to tensor

            episode_rewards += reward
            state = next_state

        total_rewards += episode_rewards
        print(f"Episode {episode+1}: Reward = {episode_rewards}")

    avg_reward = total_rewards / num_episodes
    policy_net.train()  # Revert model back to training mode if further training is required.
    return avg_reward
