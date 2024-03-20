# -*- coding: utf-8 -*-

from torch.distributions import Categorical
import torch.optim as optim
import torch
import numpy as np
import os
from tqdm import tqdm

from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts, plot_losses, epsilon_by_frame


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train_ppo(model, env, config, start_episode=0):
    num_episodes = config.get("num_episodes", 1000)
    lr = config.get("learning_rate", 1e-3)
    gamma = config.get("gamma", 0.99)
    clip_param = config.get("clip_param", 0.2)
    update_timestep = config.get("update_timestep", 2000)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = config.get("device", torch.device("cpu"))
    sequence_length = config.get("sequence_length", 4)
    num_actions = len(env.action_space)
    losses = []
    eval_vals = []
    
    timestep = 0
    episode_rewards = []
    for episode in tqdm(range(start_episode, start_episode + num_episodes)):
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
            epsilon = epsilon_by_frame(
                timestep,
                config["epsilon_start"],
                config["epsilon_final"],
                config["epsilon_decay"],
            )
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
                epsilon,
            )
            loss = update_model(
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
            if loss is not None:
                losses.append(loss)
                # reset lists
                log_probs = []
                values = []
                rewards = []
                masks = []
                states_buffer = []
        episode_rewards.append(episode_reward)
        post_episode(episode_rewards, losses, episode, model, config, env, eval_vals)

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
    epsilon=0.1  # Epsilon value for ε-greedy exploration
):
    states_buffer.append(state)
    was_random = False
    if len(states_buffer) == sequence_length:
        states_sequence = torch.stack(states_buffer, dim=0).unsqueeze(0)
        action_probs, value_estimate = model(states_sequence.to(device))
        dist = Categorical(action_probs)

        if np.random.random() < epsilon:
            # Exploration: select a random action
            action_val = np.random.choice(num_actions)
            action = torch.tensor([action_val], device=device)
            # Compute log probability for the randomly chosen action
            log_prob = dist.log_prob(action).unsqueeze(0)
            was_random = True
        else:
            # Exploitation: select the best action according to the policy
            action = dist.sample()
            action_val = action.cpu().numpy()[0]
            log_prob = dist.log_prob(action).unsqueeze(0)

    else:
        # If the buffer is not full, select a random action
        action_val = np.random.choice(num_actions)
        # No model predictions here; append zeros as placeholders or handle appropriately
        log_prob = torch.tensor([0], dtype=torch.float, device=device)  # Placeholder value
        value_estimate = torch.tensor([0], dtype=torch.float, device=device)  # Placeholder value

    next_state, reward, done = env.step(action_val)
    episode_reward += reward
    next_state = image_to_tensor(next_state, device)  # Ensure this is a function that correctly preprocesses the image
    env.record(0, "ppo", was_random, 0)  # Adjust based on your env's interface

    # Append log probability and value estimate
    log_probs.append(log_prob)
    values.append(value_estimate)
    rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
    masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

    if len(states_buffer) == sequence_length:
        states_buffer.pop(0)

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

        return loss.item()
    return None


def post_episode(episode_rewards, losses, episode, model, config, env, eval_vals):
    plot_best_attempts("./results/", 0, "PPO", episode_rewards)
    plot_losses("./results/", 0, losses)

    if (episode + 1) % config.get("checkpoint_interval", 100) == 0:
        if not os.path.exists(config.get("checkpoint", "./ppo_models")):
            os.makedirs(config.get("checkpoint", "./ppo_models"))
        torch.save(
            model.state_dict(),
            os.path.join(
                config.get("checkpoint", "./ppo_models"), f"ppo_model_ep{episode+1}.pth"
            ),
        )

    if episode % config.get("eval_frequency", 10) == 0:
        eval_reward = run_eval(model, env, config)
        eval_vals.append(eval_reward)
    if len(eval_vals) > 0:
        plot_best_attempts("./results/", 0, "PPO_eval", eval_vals)


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


def run_eval(model, eval_env, config):
    device = config.get("device", torch.device("cpu"))
    sequence_length = config.get("sequence_length", 4)
    num_eval_episodes = config.get("eval_episodes", 10)
    eval_rewards = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for _ in range(num_eval_episodes):
            state = eval_env.reset()
            state = image_to_tensor(state, device)
            states_buffer = []
            episode_reward = 0
            done = False

            while not done:
                states_buffer.append(state)
                if len(states_buffer) == sequence_length:
                    states_sequence = torch.stack(states_buffer, dim=0).unsqueeze(0)
                    action_probs, _ = model(states_sequence)
                    action = (
                        action_probs.argmax(dim=-1).cpu().numpy()[0]
                    )  # Use the action with the highest probability
                    (
                        next_state,
                        reward,
                        done,
                    ) = eval_env.step(action)
                    episode_reward += reward
                    next_state = image_to_tensor(next_state, device)
                    states_buffer.pop(0)
                else:
                    next_state, reward, done = eval_env.step(
                        np.random.choice(len(eval_env.action_space))
                    )
                    next_state = image_to_tensor(next_state, device)
                eval_env.record(0, "ppo_eval", 0, 0)
                state = next_state

            eval_rewards.append(episode_reward)

    average_reward = sum(eval_rewards) / len(eval_rewards)
    return average_reward
