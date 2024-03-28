# -*- coding: utf-8 -*-
import collections
import numpy as np
import torch.optim as optim
import torch
import os
from tqdm import tqdm
from .PPO import PPOModel
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts, plot_losses
from PoliwhiRL.environment.controller import Controller as Env


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def setup_environment_and_model(config):
    env = Env(config)
    input_dim = (1 if config["use_grayscale"] else 3, *env.screen_size())
    output_dim = len(env.action_space)
    model = PPOModel(input_dim, output_dim).to(config["device"])
    start_episode = load_latest_checkpoint(model, config["checkpoint"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    return env, model, optimizer, start_episode

@profile
def train(model, env, optimizer, config, start_episode):
    losses = []
    train_rewards = []
    replay_chance = config.get("replay_chance", 0.5)
    prev_input_sequences = []
    for episode in tqdm(
        range(start_episode, start_episode + config["num_episodes"]), desc="Training"
    ):
        state = env.reset()
        episode_rewards, saved_log_probs, saved_values, rewards, masks = (
            0,
            [],
            [],
            [],
            [],
        )
        done = False
        states_seq = collections.deque(maxlen=config["sequence_length"])
        steps_since_update = 0
        replay = np.random.rand() < replay_chance

        if replay and len(prev_input_sequences) > 0:
            min_reward = np.min(train_rewards)
            tmp_total_rewards = np.array(train_rewards)
            if min_reward < 0:
                tmp_total_rewards -= min_reward
            tmp_total_rewards += 1e-5
            chosen_index = np.random.choice(
                len(prev_input_sequences), p=tmp_total_rewards / tmp_total_rewards.sum()
            )
            replay_seq = prev_input_sequences[chosen_index]
            continue_from_point(env, replay_seq)

        while not done:
            state_tensor = image_to_tensor(state, config["device"])
            states_seq.append(state_tensor)
            if len(states_seq) < config["sequence_length"]:
                continue
            state_sequence_tensor = torch.stack(list(states_seq)).unsqueeze(0)
            action_probs, value_estimates = model(state_sequence_tensor)
            dist = torch.distributions.Categorical(action_probs[0])
            action = dist.sample()
            next_state, reward, done = env.step(action.item())
            if episode % config.get("record_frequency", 10) == 0:
                env.record(0, f"episode_{episode}")
            episode_rewards += reward
            state = next_state
            saved_log_probs.append(dist.log_prob(action).unsqueeze(0))
            saved_values.append(value_estimates)
            rewards.append(reward)
            masks.append(1.0 - done)
            steps_since_update += 1

            # Perform update and reset lists if steps_since_update reaches update_timestep
            if steps_since_update == config["update_timestep"]:
                loss = update_model(
                    optimizer,
                    saved_log_probs,
                    saved_values,
                    rewards,
                    masks,
                    config["gamma"],
                )
                losses.append(loss)
                # Reset the lists
                saved_log_probs, saved_values, rewards, masks = (
                    [],
                    [],
                    [],
                    [],
                )
                steps_since_update = 0

        if steps_since_update > 0:
            loss = update_model(
                optimizer,
                saved_log_probs,
                saved_values,
                rewards,
                masks,
                config["gamma"],
            )
            losses.append(loss)

        train_rewards.append(episode_rewards)
        prev_input_sequences.append(env.get_buttons())

        post_episode_jobs(
            model, config, episode, train_rewards, losses
        )


def update_model(optimizer, saved_log_probs, saved_values, rewards, masks, gamma):
    next_value = saved_values[-1].detach()
    returns = compute_returns(next_value, rewards, masks, gamma)

    log_probs = torch.cat(saved_log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(saved_values)

    advantages = returns - values.squeeze(-1)
    action_loss = -(log_probs * advantages).mean()
    value_loss = (returns - values.squeeze(-1)).pow(2).mean()

    optimizer.zero_grad()
    (action_loss + value_loss).backward()
    optimizer.step()
    return (action_loss + value_loss).item()


def post_episode_jobs(model, config, episode, train_rewards, losses):
    if episode % config.get("plot_every", 10) == 0:
        plot_losses("./results/", episode, losses)
        plot_best_attempts(
            "./results/",
            episode,
            f"PPO_training_{config['episode_length']}",
            train_rewards,
        )
    if episode % config.get("checkpoint_interval", 100) == 0:
        save_checkpoint(model, config["checkpoint"], episode)


def continue_from_point(env, buttons):
    env.play_button_sequence(buttons)



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
