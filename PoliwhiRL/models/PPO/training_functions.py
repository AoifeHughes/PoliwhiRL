# -*- coding: utf-8 -*-
import collections
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from .PPO import PPOModel
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts, plot_losses
from PoliwhiRL.environment import PyBoyEnvironment as Env


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def make_new_env(config):
    return Env(config)

def setup_environment_and_model(config):
    env = Env(config)
    if config["vision"]:
        height, width, channels = env.get_screen_size()
        input_dim = (channels, height, width)
    else:
        input_dim = np.prod(env.get_game_area().shape)
    output_dim = env.action_space.n
    model = PPOModel(input_dim, output_dim, config['vision']).to(config["device"])
    start_episode = load_latest_checkpoint(model, config["checkpoint"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    return env, model, optimizer, start_episode

def train(model, env, optimizer, config, start_episode):
    losses = []
    train_rewards = []
    for episode in tqdm(range(start_episode, start_episode + config["num_episodes"]), desc="Training"):
        state, _ = env.reset()
        if episode % config.get("record_frequency", 10) == 0:
            env.enable_render()
        
        episode_rewards = 0
        saved_log_probs, saved_values, rewards, masks = [], [], [], []
        saved_states, saved_actions = [], []
        done = False
        steps_since_update = 0
        
        while not done:
            if config["vision"]:
                state_tensor = image_to_tensor(state, config["device"]).unsqueeze(0).unsqueeze(0)
            else:
                state_tensor = torch.tensor(state.flatten().astype(np.float32), dtype=torch.float32, device=config["device"]).unsqueeze(0).unsqueeze(0)
            
            action_probs, value_estimates = model(state_tensor)
            dist = torch.distributions.Categorical(action_probs[0])
            action = dist.sample()
            
            next_state, reward, done, _, _ = env.step(action.item())
            
            if episode % config.get("record_frequency", 10) == 0:
                env.record(f"PPO_training_{config['episode_length']}_N_goals_{config['N_goals_target']}")
            
            episode_rewards += reward
            state = next_state
            
            saved_log_probs.append(dist.log_prob(action).unsqueeze(0))
            saved_values.append(value_estimates)
            saved_states.append(state_tensor)
            saved_actions.append(action.unsqueeze(0))
            rewards.append(reward)
            masks.append(1.0 - done)
            
            steps_since_update += 1
            
            if steps_since_update == config["update_timestep"] or done:
                states = torch.cat(saved_states)
                actions = torch.cat(saved_actions)
                loss = update_model(model, optimizer, saved_log_probs, saved_values, rewards, masks, states, actions, config["gamma"])
                losses.append(loss)
                saved_log_probs, saved_values, rewards, masks, saved_states, saved_actions = [], [], [], [], [], []
                steps_since_update = 0
        
        train_rewards.append(episode_rewards)
        post_episode_jobs(model, config, episode, train_rewards, losses)
    
    return train_rewards, losses

def update_model(model, optimizer, saved_log_probs, saved_values, rewards, masks, states, actions, gamma=0.99, epsilon=0.2, epochs=1):
    next_value = saved_values[-1].detach()
    returns = compute_returns(next_value, rewards, masks, gamma)
    old_log_probs = torch.cat(saved_log_probs).detach()
    returns = torch.cat(returns).detach()
    values = torch.cat(saved_values)
    advantages = returns - values.squeeze(-1)
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        # Detach tensors at the start of each epoch
        detached_advantages = advantages.detach()
        detached_returns = returns.detach()
        detached_old_log_probs = old_log_probs.detach()

        # Recalculate probabilities and values
        action_probs, new_values = model(states)
        dist = torch.distributions.Categorical(action_probs.squeeze(1))
        new_log_probs = dist.log_prob(actions)
        ratio = (new_log_probs - detached_old_log_probs).exp()

        # Clipped surrogate objective
        surr1 = ratio * detached_advantages
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * detached_advantages
        action_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, detached_returns)

        # Add entropy bonus
        entropy = dist.entropy().mean()
        loss = action_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def post_episode_jobs(model, config, episode, train_rewards, losses):
    if episode % config.get("plot_every", 10) == 0:
        plot_losses("./results/", f"latest_{config['episode_length']}", losses)
        plot_best_attempts(
            "./results/",
            "latest",
            f"PPO_training_{config['episode_length']}",
            train_rewards,
        )
    if episode % config.get("checkpoint_interval", 100) == 0:
        save_checkpoint(model, config["checkpoint"], episode)


def load_latest_checkpoint(model, checkpoint_dir):

    if not os.path.isdir(checkpoint_dir):
        print(
            f"No checkpoint directory found at '{checkpoint_dir}'. Starting from scratch."
        )
        return 0
    checkpoints = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("ppo_model_ep") and f.endswith(".pth")
    ]
    if not checkpoints:
        print("No checkpoints found. Starting from scratch.")
        return 0

    episodes = [int(f.split("ep")[1].split(".")[0]) for f in checkpoints]
    latest_episode = max(episodes)
    latest_checkpoint = os.path.join(
        checkpoint_dir, f"ppo_model_ep{latest_episode}.pth"
    )

    model.load_state_dict(
        torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
    )
    print(f"Loaded checkpoint from episode {latest_episode}")
    return latest_episode


def save_checkpoint(model, checkpoint_dir, episode):
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Checkpoint directory {checkpoint_dir} created.")

    checkpoint_path = os.path.join(checkpoint_dir, f"ppo_model_ep{episode}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at '{checkpoint_path}' for episode {episode}.")
