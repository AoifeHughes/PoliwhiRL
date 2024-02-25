# -*- coding: utf-8 -*-
from multiprocessing import Pool
import random
from tqdm import tqdm
import torch
from .utils import (
    optimize_model,
    beta_by_frame,
    epsilon_by_frame,
    store_experience,
)
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts
from PoliwhiRL.environment import Controller


def worker(worker_id, batch_id, config, policy_net, target_net, frame_idx_start):
    # Assuming policy_net and target_net are reconstructed within the worker from state dicts
    env = Controller(
        rom_path=config["rom_path"],
        state_path=config["state_path"],
        timeout=config["episode_length"],
        use_sight=config["sight"],
        extra_files=config["extra_files"],
        reward_locations_xy=config["reward_locations_xy"],
        scaling_factor=config["scaling_factor"],
        use_grayscale=config["use_grayscale"],
        log_path=f"./logs/double_rainbow_env_{worker_id}.json",
    )

    experiences = []
    rewards_collected = []
    td_errors = []
    frame_idx = frame_idx_start
    for episode in range(config["runs_per_worker"]):
        state = env.reset()
        state = image_to_tensor(state, config["device"])
        total_reward = 0
        done = False

        while not done:
            epsilon = epsilon_by_frame(
                frame_idx,
                config["epsilon_start"],
                config["epsilon_final"],
                config["epsilon_decay"],
            )
            beta = beta_by_frame(frame_idx, config["beta_start"], config["beta_frames"])
            action, was_random = select_action(state, epsilon, env, policy_net, config["device"])
            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, config["device"])

            store_experience(
                state,
                action,
                reward,
                next_state,
                done,
                policy_net,
                target_net,
                experiences,
                config,
                td_errors,
                beta,
            )

            if config['record']:
                env.record(epsilon, f'{batch_id}_{worker_id}', was_random)

            state = next_state
            total_reward += reward
            frame_idx += 1

        rewards_collected.append(total_reward)

    env.close()
    return experiences, rewards_collected


def select_action(state, epsilon, env, policy_net, device):
    is_random = True
    if random.random() > epsilon:
        with torch.no_grad():
            q_values = policy_net(state.unsqueeze(0).to(device))
            action = q_values.max(1)[1].view(1, 1).item()
        is_random = False
    else:
        action = env.random_move()
    return action, is_random


def run(config, policy_net, target_net, optimizer, replay_buffer):
    total_rewards = []
    total_losses = []
    frame_idx = 0
    next_target_update = frame_idx + config["target_update"]

    episodes_per_batch = config["num_workers"] * config["runs_per_worker"]
    total_batches = config["num_episodes"] // episodes_per_batch + (
        1 if config["num_episodes"] % episodes_per_batch > 0 else 0
    )

    for batch in tqdm(range(total_batches), desc="Batch Processing"):
        worker_args = [
            (i, batch, config, policy_net, target_net, frame_idx)
            for i in range(config["num_workers"])
        ]

        with Pool(processes=config["num_workers"]) as pool:
            worker_results = pool.starmap(worker, worker_args)

        all_experiences = []
        all_rewards = []
        for experiences, rewards in worker_results:
            all_experiences.extend(experiences)
            all_rewards.extend(rewards)

        for experience in tqdm(all_experiences, desc="Experience Processing"):
            state, action, reward, next_state, done, beta, td_error = experience
            replay_buffer.add(state, action, reward, next_state, done, td_error)

            loss = optimize_model(
                beta,
                policy_net,
                target_net,
                replay_buffer,
                optimizer,
                config["device"],
                config["batch_size"],
                config["gamma"],
            )
            if loss is not None:
                total_losses.append(loss)

        frame_idx += len(all_experiences)
        if frame_idx >= next_target_update:
            target_net.load_state_dict(policy_net.state_dict())
            next_target_update = frame_idx + config["target_update"]

        total_rewards.extend(all_rewards)
        
        for name in ["DoubleRainbowLatest", f"DoubleRainbow{episodes_per_batch * (batch + 1)}"]:
            plot_best_attempts(
                "./results/",
                name,
                "DoubleRainbow",
                total_rewards,
            )     


    return total_rewards, total_losses, frame_idx
