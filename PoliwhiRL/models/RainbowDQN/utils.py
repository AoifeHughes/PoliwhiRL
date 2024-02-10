# -*- coding: utf-8 -*-
import torch
import os
import numpy as np


def beta_by_frame(frame_idx, beta_start, beta_frames):
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


def epsilon_by_frame(frame_idx, epsilon_start, epsilon_final, epsilon_decay):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1.0 * frame_idx / epsilon_decay
    )

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


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves the current state of training."""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)


def load_checkpoint(filename="checkpoint.pth.tar", device="cpu"):
    """Loads the checkpoint and returns the state."""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        print(f"Checkpoint loaded from {filename}")
        return checkpoint
    else:
        print(f"Checkpoint file not found: {filename}")
        return None
