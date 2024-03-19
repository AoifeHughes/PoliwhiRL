# -*- coding: utf-8 -*-
import math
import random
import torch
import os
import numpy as np


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
    action_sequence_tensor = torch.tensor(action_sequence, device=config["device"], dtype=torch.long)
    reward_sequence_tensor = torch.tensor(reward_sequence, device=config["device"], dtype=torch.float)
    next_state_sequence_tensor = torch.stack(next_state_sequence).to(config["device"])
    done_sequence_tensor = torch.tensor(done_sequence, device=config["device"], dtype=torch.bool)
    
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
        state_sequence_tensor,
        action_sequence_tensor,
        reward_sequence_tensor,
        next_state_sequence_tensor,
        done_sequence_tensor,
        td_error
    )

def compute_td_error_sequence(
    state_sequence_tensor, action_sequence_tensor, reward_sequence_tensor, next_state_sequence_tensor, done_sequence_tensor, policy_net, target_net, gamma, device
):
    
    current_q_values_all = policy_net(state_sequence_tensor)
    current_q_values = current_q_values_all.gather(1, action_sequence_tensor.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_q_values_all = target_net(next_state_sequence_tensor)
        next_q_values = next_q_values_all.max(1)[0].detach()
        next_q_values[done_sequence_tensor] = 0.0  # Zero-out values for done states
        expected_q_values = reward_sequence_tensor + (gamma * next_q_values)

    td_error = (expected_q_values - current_q_values).abs().mean()
    return td_error


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
    # Check if the replay buffer has enough samples
    if len(replay_buffer) < batch_size:
        print("Not enough samples in the replay buffer.")
        return
    
    # Sample a batch of sequences from the replay buffer
    states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)
    
    # No need to reshape since we expect the replay buffer to provide the correct shape
    # and the model's forward method has been adjusted to handle it directly
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    rewards = torch.stack(rewards).to(device)
    next_states = torch.stack(next_states).to(device)
    dones = torch.stack(dones).to(device)
    weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1).to(device)

    # Current Q values as predicted by the policy net
    current_q_values = policy_net(states).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    # Next Q values as predicted by the target net for all next states
    with torch.no_grad():
        next_q_values = target_net(next_states).max(-1)[0].detach()
        # Apply mask for dones. Since dones are provided as a tensor of the same batch and sequence shape,
        # use it to zero out the next Q values for done states
        next_q_values[dones] = 0.0
        expected_q_values = rewards + (gamma * next_q_values)

    # Compute the loss, taking into account the importance sampling weights
    loss = ((current_q_values - expected_q_values.detach()) ** 2 * weights).mean()
    
    # Prioritize samples based on the TD error, adding a small value to ensure no zero priorities
    prios = loss.detach() + 1e-5
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update priorities in the replay buffer using the squared TD errors
    replay_buffer.update_priorities(indices, prios.squeeze().cpu().numpy())

    return loss.item()

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


def store_experience(
    state,
    action,
    reward,
    next_state,
    done,
    policy_net,
    target_net,
    replay_buffer,
    config,
    td_errors,
    beta=None,
):
    # Assuming state, next_state are already tensors and correctly placed on the device
    action_tensor = torch.tensor([action], device=config["device"], dtype=torch.long)
    reward_tensor = torch.tensor([reward], device=config["device"], dtype=torch.float)
    done_tensor = torch.tensor([done], device=config["device"], dtype=torch.bool)

    # Compute TD error with reduced redundant operations
    td_error = compute_td_error(
        state.unsqueeze(0),  # Adding batch dimension here, assuming state is a tensor
        action_tensor.unsqueeze(0),  # Adding batch dimension
        reward_tensor,
        next_state.unsqueeze(0),  # Adding batch dimension
        done_tensor,
        policy_net,
        target_net,
        config["gamma"],
    )
    td_errors.append(td_error)

    return replay_buffer.add(
        state, action_tensor, reward_tensor, next_state, done_tensor, td_error
    )


def compute_td_error(
    state, action, reward, next_state, done, policy_net, target_net, gamma=0.99
):
    """
    Optimized TD error computation to minimize redundant tensor operations and ensure efficiency.
    """
    # Assuming state, action, reward, next_state, and done are correctly shaped tensors on the correct device
    current_q_values = policy_net(state).gather(1, action).squeeze(-1)

    # Compute next Q values from target network without unnecessary tensor operations
    with torch.no_grad():
        next_state_values = target_net(next_state).max(1)[0].detach()
        next_state_values[done] = 0.0  # Zero-out terminal states
        expected_q_values = reward + gamma * next_state_values

    td_error = (expected_q_values - current_q_values).abs()
    return td_error.item()  # Keep as scalar if necessary for external use


def optimize_model(
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

    # Directly convert tuples to tensors without np.array conversion
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    rewards = torch.stack(rewards).to(device)
    next_states = torch.stack(next_states).to(device)
    dones = torch.stack(dones).to(device)
    weights = torch.FloatTensor(weights).unsqueeze(-1).to(device)

    # Current Q values
    current_q_values = policy_net(states).gather(1, actions)

    # Next Q values based on the action chosen by policy_net
    next_q_values = policy_net(next_states).detach()
    _, best_actions = next_q_values.max(1, keepdim=True)

    # Next Q values from target_net for actions chosen by policy_net
    next_q_values_target = target_net(next_states).detach().gather(1, best_actions)

    # Expected Q values
    expected_q_values = rewards + (gamma * next_q_values_target * (~dones)).float()

    # Compute the loss
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


def select_action_hybrid(
    state, policy_net, config, frame_idx, action_counts, num_actions, epsilon
):
    # Decide to take a random action with probability epsilon
    if random.random() < epsilon:
        return random.randrange(num_actions), None  # Return a random action

    with torch.no_grad():
        # Obtain Q-values from the policy network for the current state
        q_values = policy_net(state).cpu().numpy()[0]

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
