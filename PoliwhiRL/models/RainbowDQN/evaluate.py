import torch
import random
from PoliwhiRL.utils import image_to_tensor

def evaluate_model(config, env, policy_net):
    policy_net.eval()  # Ensure the model is in evaluation mode.
    total_rewards = 0
    num_episodes = config.get("eval_episodes", 10)  # Default to 10 episodes for evaluation if not specified.
    sequence_length = config.get("sequence_length", 4)  # Default sequence length if not specified in config.

    for episode in range(num_episodes):
        state = env.reset()
        state = image_to_tensor(state, config["device"])  # Convert state to tensor
        state_sequence = [torch.zeros_like(state) for _ in range(sequence_length)] 
        episode_rewards = 0
        done = False

        while not done:
            state_sequence.append(state)  # Add the latest state to the sequence
            state_sequence.pop(0)  # Remove the oldest state

            if len(state_sequence) < sequence_length:
                # If the buffer hasn't reached the full length, select a random action
                action = random.randint(1, 8)  # Assuming action space is 1-8
            else:
                # Once the buffer is full, prepare the sequence for the model
                sequence_tensor = torch.stack(state_sequence).unsqueeze(0).to(config["device"])
                with torch.no_grad():  # No need to track gradients during evaluation
                    q_values = policy_net(sequence_tensor)
                    action = torch.argmax(q_values[-1]).item()  # Select the action with the highest Q-value

            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, config["device"])

            env.record(0, "rdqn_eval", False, 0)
            episode_rewards += reward
            state = next_state  # Update the current state to the next state

        total_rewards += episode_rewards

    avg_reward = total_rewards / num_episodes
    policy_net.train()  # Revert model back to training mode if further training is required.
    return avg_reward
