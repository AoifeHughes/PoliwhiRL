# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import numpy as np
import random
from PoliwhiRL.models.RainbowDQN.ReplayBuffer import PrioritizedReplayBuffer
from PoliwhiRL.environment.controls import Controller
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts
from tqdm import tqdm
from PoliwhiRL.models.RainbowDQN.NoisyLinear import NoisyLinear
import time
import json


class RainbowDQN(nn.Module):
    def __init__(self, input_dim, num_actions, device, atom_size=51, Vmin=-10, Vmax=10):
        super(RainbowDQN, self).__init__()
        self.num_actions = num_actions
        self.atom_size = atom_size
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.support = (
            torch.linspace(Vmin, Vmax, atom_size).view(1, 1, atom_size).to(device)
        )

        # Define network layers
        self.feature_layer = nn.Sequential(
            nn.Conv2d(
                input_dim[0], 32, kernel_size=8, stride=4, padding=1
            ),  # Adjusted for input dimensions
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_input_dim = self.feature_size(input_dim)

        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, atom_size)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * atom_size),
        )

    def forward(self, x):
        dist = self.get_distribution(x)
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values

    def get_distribution(self, x):
        x = self.feature_layer(x)
        value = self.value_stream(x).view(-1, 1, self.atom_size)
        advantage = self.advantage_stream(x).view(-1, self.num_actions, self.atom_size)
        advantage_mean = advantage.mean(1, keepdim=True)
        dist = value + advantage - advantage_mean
        dist = F.softmax(dist, dim=-1)
        dist = dist.clamp(min=1e-3)  # Avoid zeros
        return dist

    def feature_size(self, input_dim):
        return self.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1)

    def reset_noise(self):
        """Reset all noisy layers"""
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


def run(
    rom_path,
    state_path,
    episode_length,
    device,
    num_episodes,
    batch_size,
    checkpoint_path="rainbow_checkpoint.pth",
):
    start_time = time.time()  # For computational efficiency tracking
    env = Controller(
        rom_path, state_path, timeout=episode_length, log_path="./logs/rainbow_env.json"
    )
    gamma = 0.99
    alpha = 0.6
    beta_start = 0.4
    beta_frames = 1000
    frame_idx = 0
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000
    learning_rate = 1e-4
    capacity = 10000
    update_target_every = 1000
    losses = []
    epsilon_values = []  # Tracking epsilon values for exploration metrics
    beta_values = []  # For priority buffer metrics
    td_errors = []  # For DQN metrics
    rewards = []
    screen_size = env.screen_size()
    input_shape = (3, int(screen_size[0]), int(screen_size[1]))
    policy_net = RainbowDQN(input_shape, len(env.action_space), device).to(device)
    target_net = RainbowDQN(input_shape, len(env.action_space), device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = PrioritizedReplayBuffer(capacity, alpha)

    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint is not None:
        start_episode = checkpoint["episode"] + 1
        frame_idx = checkpoint["frame_idx"]
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        target_net.load_state_dict(checkpoint["target_net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    else:
        start_episode = 0

    for episode in tqdm(range(start_episode, start_episode + num_episodes)):
        state = env.reset()
        state = image_to_tensor(state, device)

        total_reward = 0
        ep_len = 0
        while True:
            frame_idx += 1
            #frame_loc_idx = env.get_frames_in_current_location()
            epsilon = epsilon_by_frame(
                frame_idx, epsilon_start, epsilon_final, epsilon_decay
            )
            epsilon_values.append(epsilon)  # Log epsilon value
            beta = beta_by_frame(frame_idx, beta_start, beta_frames)
            beta_values.append(beta)  # Log beta value

            if random.random() > epsilon:
                with torch.no_grad():
                    state_t = state.unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    action = q_values.max(1)[1].item()
            else:
                action = env.random_move()

            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, device)
            action = torch.tensor([action], device=device)
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)
            td_error = compute_td_error(
                (state, action, reward, next_state, done),
                policy_net,
                target_net,
                device,
                gamma,
            )
            td_errors.append(td_error)  # Log TD error
            replay_buffer.add(state, action, reward, next_state, done, error=td_error)
            state = next_state
            total_reward += reward.item()

            loss = optimize_model(
                beta,
                policy_net,
                target_net,
                replay_buffer,
                optimizer,
                device,
                batch_size,
                gamma,
            )
            if loss is not None:
                losses.append(loss)

            if frame_idx % update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if episode % 100 == 0 or episode == num_episodes - 1:
                env.record(episode, 1, "Rainbow")
            if done:
                break
            ep_len += 1
        rewards.append(total_reward)
        if episode % 100 == 0 and episode > 0:
            plot_best_attempts("./results/", '', f"Rainbow DQN_latest", rewards)


    total_time = time.time() - start_time  # Total training time
    env.close()

    # Prepare logging data
    log_data = {
        "total_time": total_time,
        "average_reward": sum(rewards) / len(rewards),
        "losses": losses,
        "epsilon_values": epsilon_values,
        "beta_values": beta_values,
        "td_errors": td_errors,
    }

    # Save logged data to file
    # check folder exists and create
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    with open("./logs/training_log.json", "w") as outfile:
        json.dump(log_data, outfile, indent=4)
    print("Training log saved to ./logs/training_log.json")


def beta_by_frame(frame_idx, beta_start, beta_frames):
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


def epsilon_by_frame(frame_idx, epsilon_start, epsilon_final, epsilon_decay):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1.0 * frame_idx / epsilon_decay
    )


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
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename="checkpoint.pth.tar", device="cpu"):
    """Loads the checkpoint and returns the state."""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        print(f"Checkpoint loaded from {filename}")
        return checkpoint
    else:
        print(f"Checkpoint file not found: {filename}")
        return None
