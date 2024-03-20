# -*- coding: utf-8 -*-
import math
import random
import torch
import os
import numpy as np

from PoliwhiRL.utils.utils import image_to_tensor


def beta_by_frame(frame_idx, beta_start, beta_frames):
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


def epsilon_by_frame(frame_idx, epsilon_start, epsilon_final, epsilon_decay):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1.0 * frame_idx / epsilon_decay
    )


def store_experience_sequence(
    state_sequence,
    action_sequence,
    reward_sequence,
    next_state_sequence,
    done_sequence,
    policy_net,
    target_net,
    replay_buffer,
    config,
    td_errors,
    beta=None,
):
    # Convert sequences to tensors with the appropriate device and dtype
    state_sequence_tensor = torch.stack(state_sequence).to(config["device"])
    action_sequence_tensor = torch.tensor(
        action_sequence, device=config["device"], dtype=torch.long
    )
    reward_sequence_tensor = torch.tensor(
        reward_sequence, device=config["device"], dtype=torch.float
    )
    next_state_sequence_tensor = torch.stack(next_state_sequence).to(config["device"])
    done_sequence_tensor = torch.tensor(
        done_sequence, device=config["device"], dtype=torch.bool
    )

    # Compute TD error for the entire sequence
    td_error = compute_td_error_sequence(
        state_sequence_tensor,
        action_sequence_tensor,
        reward_sequence_tensor,
        next_state_sequence_tensor,
        done_sequence_tensor,
        policy_net,
        target_net,
        config["gamma"],
        config["device"],
    )
    td_errors.append(td_error)

    # Add the entire sequence to the replay buffer
    # The sequences are now tensors, ready for efficient storage and retrieval
    return replay_buffer.add(
        (
            state_sequence_tensor,
            action_sequence_tensor,
            reward_sequence_tensor,
            next_state_sequence_tensor,
            done_sequence_tensor,
        ),
        td_error,
    )


def compute_td_error_sequence(
    state_sequence,
    action_sequence,
    reward_sequence,
    next_state_sequence,
    done_sequence,
    policy_net,
    target_net,
    gamma,
    device,
):

    state_sequence = state_sequence.to(device).unsqueeze(0)
    next_state_sequence = next_state_sequence.to(device).unsqueeze(0)
    action_sequence = action_sequence.to(device).unsqueeze(-1)
    reward_sequence = reward_sequence.to(device).unsqueeze(-1)
    done_sequence = done_sequence.to(device).unsqueeze(-1)

    q_values = policy_net(state_sequence)

    state_action_values = q_values.gather(1, action_sequence)

    with torch.no_grad():
        next_state_values = target_net(next_state_sequence).max(1)[0].detach()
        next_state_values = next_state_values.unsqueeze(-1)

    expected_state_action_values = reward_sequence + (
        gamma * next_state_values * (~done_sequence).float()
    )

    td_errors = (expected_state_action_values - state_action_values).squeeze(-1)
    return td_errors.abs().mean().item()


def optimize_model_sequence(
    beta,
    policy_net,
    target_net,
    replay_buffer,
    optimizer,
    device,
    batch_size=32,
    gamma=0.99,
):
    if len(replay_buffer) < batch_size:
        return
    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        indices,
        weights,
    ) = replay_buffer.sample(batch_size, beta)

    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    next_states = torch.stack(next_states)
    dones = torch.stack(dones)
    weights = torch.FloatTensor(weights).unsqueeze(-1).to(device)

    current_q_values = policy_net(states).gather(1, actions)
    next_q_values = policy_net(next_states).detach()

    _, best_actions = next_q_values.max(1, keepdim=True)

    next_q_values_target = (
        target_net(next_states).detach().gather(1, best_actions).view(batch_size, -1)
    )

    expected_q_values = rewards + (gamma * next_q_values_target * (~dones)).float()
    loss = (current_q_values - expected_q_values).pow(2) * weights
    prios = loss + 1e-5  # Avoid zero priority
    loss = loss.mean()

    # Perform optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update priorities in the buffer
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

    return loss.item()  # Optional: return the loss value for monitoring


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

# TODO Update this function to use the new environment interface
def select_action_hybrid(
    states, policy_net, config, frame_idx, action_counts, num_actions, epsilon
):
    # Decide to take a random action with probability epsilon
    if random.random() < epsilon:
        return random.randrange(num_actions), None  # Return a random action

    with torch.no_grad():
        # Obtain Q-values from the policy network for the current state
        q_values = (
            policy_net( torch.stack(states).unsqueeze(0).to(config["device"]) )
        )
        action = torch.argmax(q_values[-1]).item() 

    exploration_rate = np.sqrt(
        2 * math.log(frame_idx + 1) / (action_counts + 1)
    )  # Avoid division by zero
    hybrid_values = (
        q_values + exploration_rate
    )  # Combine Q-values with exploration bonus

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


def populate_replay_buffer(
    config, env, replay_buffer, policy_net, target_net, td_errors
):
    policy_net.reset_noise()
    state_sequence = []
    action_sequence = []
    reward_sequence = []
    next_state_sequence = []
    done_sequence = []
    state = env.reset()
    env.extend_timeout(250)
    state = image_to_tensor(state, config["device"])
    sequence_length = config.get("sequence_length", 4)
    num_actions = len(env.action_space)
    done = False
    while not done:
        action = np.random.choice(num_actions)
        next_state, reward, done = env.step(action)
        next_state = image_to_tensor(next_state, config["device"])
        state_sequence.append(state)
        action_sequence.append(action)
        reward_sequence.append(reward)
        next_state_sequence.append(next_state)
        done_sequence.append(done)
        if len(state_sequence) == sequence_length:
            store_experience_sequence(
                state_sequence,
                action_sequence,
                reward_sequence,
                next_state_sequence,
                done_sequence,
                policy_net,
                target_net,
                replay_buffer,
                config,
                td_errors,
            )
            state_sequence = []
            action_sequence = []
            reward_sequence = []
            next_state_sequence = []
            done_sequence = []
