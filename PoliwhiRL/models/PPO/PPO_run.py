# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from PoliwhiRL.environment.controller import Controller
from PoliwhiRL.models.PPO.training_functions import compute_returns
from .PPO import PPOModel
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts
import os


def train_ppo(model, env, config, start_episode=0):
    num_episodes = config.get("num_episodes", 1000)
    lr = config.get("learning_rate", 1e-3)
    gamma = config.get("gamma", 0.99)
    clip_param = config.get("clip_param", 0.2)
    update_timestep = config.get("update_timestep", 2000)
    save_dir = config.get("checkpoint", "./ppo_models")
    save_freq = config.get("checkpoint_interval", 100)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = config.get("device", torch.device("cpu"))
    sequence_length = config.get("sequence_length", 4)
    num_actions = len(env.action_space)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestep = 0
    episode_rewards = []
    for episode in tqdm(range(start_episode + 1, start_episode + num_episodes)):
        episode_reward = 0
        log_probs = []
        values = []
        rewards = []
        masks = []
        states_buffer = []

        state = env.reset()
        state = image_to_tensor(state, device)
        done = False

        while not done:
            timestep += 1
            state, episode_reward, done = run_episode_step(
                model,
                env,
                state,
                states_buffer,
                sequence_length,
                device,
                num_actions,
                log_probs,
                values,
                rewards,
                masks,
                episode_reward,
            )
            update_model(
                model,
                optimizer,
                log_probs,
                values,
                rewards,
                masks,
                states_buffer,
                sequence_length,
                timestep,
                update_timestep,
                done,
                gamma,
                clip_param,
            )
        episode_rewards.append(episode_reward)
        post_episode(episode_rewards, episode, save_freq, model, save_dir)

    return episode_rewards


def run_episode_step(
    model,
    env,
    state,
    states_buffer,
    sequence_length,
    device,
    num_actions,
    log_probs,
    values,
    rewards,
    masks,
    episode_reward,
):
    states_buffer.append(state)

    if len(states_buffer) == sequence_length:
        states_sequence = torch.stack(states_buffer, dim=0).unsqueeze(0)
        action_probs, value = model(states_sequence)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_val = action.cpu().numpy()[0]
        next_state, reward, done = env.step(action_val)
        episode_reward += reward
        next_state = image_to_tensor(next_state, device)
        env.record(0, "ppo", False, 0)
        log_prob = dist.log_prob(action).unsqueeze(0)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
        masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))
        states_buffer.pop(0)
    else:
        next_state, reward, done = env.step(np.random.choice(num_actions))
        next_state = image_to_tensor(next_state, device)
        env.record(0, "ppo", True, 0)

    return next_state, episode_reward, done


def update_model(
    model,
    optimizer,
    log_probs,
    values,
    rewards,
    masks,
    states_buffer,
    sequence_length,
    timestep,
    update_timestep,
    done,
    gamma,
    clip_param,
):
    if timestep % update_timestep == 0 or done:
        if len(states_buffer) > 0:
            padded_sequence = states_buffer + [states_buffer[-1]] * (
                sequence_length - len(states_buffer)
            )
            states_sequence = torch.stack(padded_sequence, dim=0).unsqueeze(0)
            _, next_value = model(states_sequence)
        else:
            _, next_value = model(states_sequence)

        returns = compute_returns(next_value, rewards, masks, gamma)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantage.detach()
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage.detach()
        )
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_probs = []
        values = []
        rewards = []
        masks = []
        states_buffer = []


def post_episode(episode_rewards, episode, save_freq, model, save_dir):
    plot_best_attempts("./results/", 0, "PPO", episode_rewards)
    if (episode + 1) % save_freq == 0:
        torch.save(
            model.state_dict(), os.path.join(save_dir, f"ppo_model_ep{episode+1}.pth")
        )


import os
import torch


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


def setup_and_train_ppo(config):
    env = Controller(config)
    input_dim = (1 if config["use_grayscale"] else 3, *env.screen_size())
    output_dim = len(env.action_space)
    model = PPOModel(input_dim, output_dim).to(config["device"])
    start_episode = load_latest_checkpoint(model, config["checkpoint"])
    train_ppo(model, env, config, start_episode)
