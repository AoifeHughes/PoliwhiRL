# -*- coding: utf-8 -*-
import math
import random
import torch
import os
import numpy as np


def beta_by_frame(frame_idx, beta_start, beta_frames):
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


def epsilon_by_frame(frame_idx, epsilon_start, epsilon_final, epsilon_decay):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1.0 * frame_idx / epsilon_decay
    )


def epsilon_by_frame_with_reward(
    frame_idx,
    epsilon_start,
    epsilon_final,
    epsilon_decay,
    average_reward,
    reward_threshold,
    reward_sensitivity,
):
    if average_reward > reward_threshold:
        decay_adjustment = reward_sensitivity
    else:
        decay_adjustment = -reward_sensitivity
    adjusted_epsilon_decay = epsilon_decay + decay_adjustment * epsilon_decay
    adjusted_epsilon_decay = max(adjusted_epsilon_decay, 1)
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1.0 * frame_idx / adjusted_epsilon_decay
    )

    return epsilon


def epsilon_by_frame_cyclic(frame_idx, epsilon_start, epsilon_final, epsilon_decay):
    # Modulate frame_idx to create a cyclical effect
    modulated_frame_idx = np.cos(frame_idx * (2 * np.pi / epsilon_decay))

    # Scale and shift the modulated index so it oscillates between 0 and 1
    normalized_frame_idx = (modulated_frame_idx + 1) / 2

    # Calculate epsilon using a modified approach that incorporates the cyclic behavior
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1.0 * normalized_frame_idx * frame_idx / epsilon_decay
    )

    return epsilon



def save_checkpoint(
    config,
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    rewards,
    filename=None,
    episodes=None,
    frames=None,
):
    # Use filename from config if not provided
    if filename is None:
        filename = config["checkpoint"]

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Prepare the state to be saved
    state = {
        "episode": episodes,
        "frame_idx": frames,
        "policy_net_state_dict": policy_net.state_dict(),
        "target_net_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_buffer_state_dict": replay_buffer.state_dict(),
        "rewards": rewards,
    }

    # Save the state to file
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(config):
    filename = config.get("checkpoint", "checkpoint.pth.tar")
    device = config.get("device", "cpu")

    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        print(f"Checkpoint loaded from {filename}")

        # Update the config with loaded checkpoint info
        config.update(
            {
                "start_episode": checkpoint.get("episode", 0) + 1,
                "frame_idx": checkpoint.get("frame_idx", 0),
                "policy_net_state_dict": checkpoint.get("policy_net_state_dict"),
                "target_net_state_dict": checkpoint.get("target_net_state_dict"),
                "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
                "replay_buffer_state_dict": checkpoint.get("replay_buffer_state_dict"),
                "frames_in_loc": checkpoint.get("frames_in_loc", {}),
                "rewards": checkpoint.get("rewards", []),
            }
        )

        return checkpoint
    else:
        print(f"Checkpoint file not found: {filename}")
        return None


def select_action_hybrid(state, policy_net, config, frame_idx, action_counts, num_actions, epsilon):
    # Decide to take a random action with probability epsilon
    if random.random() < epsilon:
        return random.randrange(num_actions), None  # Return a random action

    with torch.no_grad():
        # Expand the single state to a batch by replicating it
        state_batch = state.unsqueeze(0).repeat((config['batch_size'], 1, 1, 1)).to(config["device"])
        # Obtain Q-values from the policy network for the current (dummy) batch
        q_values_batch = policy_net(state_batch).cpu().numpy()
        # Use the Q-values of the first state in the batch, as all are identical
        q_values = q_values_batch[0]

    exploration_rate = np.sqrt(
        2 * math.log(frame_idx + 1) / (action_counts + 1)
    )  # Avoid division by zero
    hybrid_values = q_values + exploration_rate  # Combine Q-values with exploration bonus

    for action in range(num_actions):
        if action_counts[action] == 0:
            # Ensure untried actions are considered
            hybrid_values[action] += np.inf

    action = np.argmax(hybrid_values)
    action_counts[action] += 1  # Update the counts for the selected action

    return action, q_values[action]
def select_action_eval(state, policy_net, config):

    with torch.no_grad():
        # Obtain Q-values from the policy network for the current state
        q_values = policy_net(state.unsqueeze(0).to(config["device"])).cpu().numpy()[0]

    # Select the action with the highest Q-value
    action = np.argmax(q_values)

    return action, q_values[action]
