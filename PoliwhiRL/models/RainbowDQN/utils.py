# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
from .replaybuffer import PrioritizedReplayBuffer
import torch.nn.functional as F

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
    ) = replay_buffer.sample(batch_size, beta, device=device)

    if states.dim() == 5:
        states = states.squeeze(1)  # Remove the extra dimension
    if next_states.dim() == 5:
        next_states = next_states.squeeze(1)  # Remove the extra dimension

    weights = weights.unsqueeze(-1)  # Ensure weights are correctly shaped for the loss calculation

    # Current Q values
    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(-1)

    # Next Q values based on the action chosen by policy_net
    next_q_values = policy_net(next_states).detach()
    _, best_actions = next_q_values.max(1, keepdim=True)

    # Next Q values from target_net for actions chosen by policy_net
    next_q_values_target = target_net(next_states).detach().gather(1, best_actions).squeeze(-1)

    # Expected Q values
    expected_q_values = rewards + (gamma * next_q_values_target * (~dones))

    # Compute the loss using SmoothL1Loss for stability
    loss = (weights * F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')).mean()

    # Calculate priorities for experience replay
    prios = loss.detach() + 1e-5  # Ensure positive priorities with a small constant

    # Perform optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update priorities in the buffer
    replay_buffer.update_priorities(indices, prios.cpu().numpy())


    return loss.item()  # Optional: return the loss value for monitoring



def save_checkpoint(
    config,
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    rewards,
    filename=None,
    epsilons_by_location=None,
):
    # Use filename from config if not provided
    if filename is None:
        filename = config["checkpoint"]

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Prepare the state to be saved
    state = {
        "episode": config.get("num_episodes", 0),
        "frame_idx": config.get("frame_idx", 0),
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
                "epsilon_by_location": checkpoint.get("epsilon_by_location", {}),
            }
        )

        return checkpoint
    else:
        print(f"Checkpoint file not found: {filename}")
        return None


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
    beta,
    n_step_buffer=None  # Add n_step_buffer as a parameter
):
    """
    Stores the experience in the replay buffer and computes TD error.
    """
    action_tensor = torch.tensor([action], device=config["device"], dtype=torch.long)
    reward_tensor = torch.tensor([reward], device=config["device"], dtype=torch.float)
    done_tensor = torch.tensor([done], device=config["device"], dtype=torch.bool)
    td_error = compute_td_error(
        (state, action_tensor, reward_tensor, next_state, done_tensor),
        policy_net,
        target_net,
        config["device"],
        config["gamma"],
    )
    td_errors.append(td_error)

    if n_step_buffer is not None:
        # If n_step_buffer is provided, use it to store n-step experiences
        add_n_step_experience(
            state, action, reward, next_state, done, replay_buffer, n_step_buffer, policy_net, target_net, config
        )
    else:
        # Otherwise, add the experience as usual
        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            replay_buffer.add(
                state, action_tensor, reward_tensor, next_state, done_tensor, td_error
            )
        else:
            replay_buffer.append(
                (
                    state,
                    action_tensor,
                    reward_tensor,
                    next_state,
                    done_tensor,
                    beta,  
                    td_error,
                )
            )


def add_n_step_experience(
    state, action, reward, next_state, done, 
    replay_buffer, n_step_buffer, policy_net, target_net, config
):
    """
    Adds an experience to the n-step buffer and updates the replay buffer with n-step experiences.
    """
    # Append the current experience to the n_step_buffer
    n_step_buffer.append((state, action, reward, next_state, done))
    gamma = config['gamma']
    # If the buffer has enough experiences, calculate the n-step return
    if len(n_step_buffer) >= config['n_steps']:
        R = 0
        for i in range(config['n_steps']):
            _, _, r, _, _ = n_step_buffer[i]
            R += (gamma ** i) * r
        
        n_step_state, n_step_action, _, _, _ = n_step_buffer.popleft()
        
        # The next state and done flag from the last experience in the n-step sequence
        _, _, _, n_step_next_state, n_step_done = n_step_buffer[-1]
        
        # Preparing tensors for TD error computation
        n_step_state_tensor = n_step_state.to(config["device"]).unsqueeze(0)
        n_step_next_state_tensor = n_step_next_state.to(config["device"]).unsqueeze(0)
        n_step_action_tensor = torch.tensor([n_step_action], device=config["device"], dtype=torch.long)
        n_step_reward_tensor = torch.tensor([R], device=config["device"], dtype=torch.float)
        n_step_done_tensor = torch.tensor([n_step_done], device=config["device"], dtype=torch.bool)

        # Compute TD error for the n-step experience using a modified compute_td_error to handle n-step rewards
        td_error = compute_n_step_td_error(
            (n_step_state_tensor, n_step_action_tensor, n_step_reward_tensor, n_step_next_state_tensor, n_step_done_tensor),
            policy_net, target_net, config['device'], gamma ** config['n_steps']
        )

        # Add the n-step experience to the replay buffer
        replay_buffer.add(
            n_step_state_tensor, n_step_action_tensor, n_step_reward_tensor, n_step_next_state_tensor, n_step_done_tensor, td_error
        )

def compute_n_step_td_error(experience, policy_net, target_net, device, effective_gamma):
    state, action, reward, next_state, done = experience

    # Compute current Q values: Q(s, a)
    current_q_values = policy_net(state).gather(1, action.unsqueeze(-1)).squeeze(-1)

    # Compute next Q values from target network
    with torch.no_grad():
        next_state_values = target_net(next_state).max(1)[0].detach()
        next_state_values[done] = 0.0  # Zero-out terminal states
        expected_q_values = reward + effective_gamma * next_state_values

    # TD error
    td_error = (expected_q_values - current_q_values).abs()
    return td_error.item()  # Return absolute TD error as scalar



def compute_td_error(experience, policy_net, target_net, device, gamma=0.99):
    state, action, reward, next_state, done = experience

    # Ensure tensors are on the correct device and add batch dimension since dealing with single experience
    state = state.to(device).unsqueeze(0)  # Add batch dimension
    next_state = next_state.to(device).unsqueeze(0)  # Add batch dimension
    action = torch.tensor([action], device=device, dtype=torch.long)
    reward = torch.tensor([reward], device=device, dtype=torch.float)
    done = torch.tensor([done], device=device, dtype=torch.bool)

    # Compute current Q values: Q(s, a)
    current_q_values = policy_net(state).gather(1, action.unsqueeze(-1)).squeeze(-1)

    # Compute next Q values from target network
    with torch.no_grad():
        next_state_values = target_net(next_state).max(1)[0].detach()
        next_state_values[done] = 0.0  # Zero-out terminal states
        expected_q_values = reward + gamma * next_state_values

    # TD error
    td_error = (expected_q_values - current_q_values).abs()
    return td_error.item()  # Return absolute TD error as scalar


