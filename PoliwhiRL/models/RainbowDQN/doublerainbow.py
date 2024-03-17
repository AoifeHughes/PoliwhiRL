# -*- coding: utf-8 -*-
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from .utils import (
    optimize_model,
    beta_by_frame,
    epsilon_by_frame,
    store_experience,
    select_action_hybrid,
)
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts
from PoliwhiRL.environment import Controller


def worker(
    worker_id,
    batch_id,
    config,
    policy_net,
    target_net,
    frame_idx_start,
    action_counts,
    action_rewards,
):

    env = Controller(config)
    experiences = []
    rewards_collected = []
    td_errors = []
    frame_idx = frame_idx_start

    for episode in range(config["runs_per_worker"]):
        state = env.reset()
        policy_net.reset_noise()
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

            action, was_random = select_action_hybrid(
                state,
                policy_net,
                config,
                frame_idx,
                action_counts,
                len(action_counts),
                epsilon,
            )
            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, config["device"])
            action_rewards[action] += reward

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

            if config["record"]:
                env.record(epsilon, f"{batch_id}_{worker_id}", was_random)

            state = next_state
            total_reward += reward
            frame_idx += 1

        rewards_collected.append(total_reward)

    env.close()
    return experiences, rewards_collected, action_counts, action_rewards


def run(config, policy_net, target_net, optimizer, replay_buffer, num_actions):
    total_rewards = []
    total_losses = []
    total_beta_values = []
    total_td_errors = []

    frame_idx = 0
    next_target_update = frame_idx + config["target_update"]

    action_counts = np.zeros(num_actions, dtype=int)
    action_rewards = np.zeros(num_actions, dtype=float)

    episodes_per_batch = config["num_workers"] * config["runs_per_worker"]
    total_batches = config["num_episodes"] // episodes_per_batch + (
        1 if config["num_episodes"] % episodes_per_batch > 0 else 0
    )

    for batch in tqdm(range(total_batches), desc="Batch Processing"):
        worker_args = [
            (
                i,
                batch,
                config,
                policy_net,
                target_net,
                frame_idx,
                action_counts.copy(),
                action_rewards.copy(),
            )
            for i in range(config["num_workers"])
        ]

        with Pool(processes=config["num_workers"]) as pool:
            worker_results = pool.starmap(worker, worker_args)

        all_experiences = []
        all_rewards = []
        all_beta_values = []
        all_td_errors = []
        action_counts = np.zeros(num_actions, dtype=int)
        action_rewards = np.zeros(num_actions, dtype=float)

        for (
            experiences,
            rewards,
            worker_action_counts,
            worker_action_rewards,
        ) in worker_results:
            all_experiences.extend(experiences)
            all_rewards.extend(rewards)
            action_counts += worker_action_counts
            action_rewards += worker_action_rewards

        for experience in all_experiences:
            state, action, reward, next_state, done, beta, td_error = experience
            all_td_errors.append(td_error)
            replay_buffer.add(state, action, reward, next_state, done, td_error)
            all_beta_values.append(beta)
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
        total_beta_values.extend(all_beta_values)
        total_td_errors.extend(all_td_errors)

        for name in [
            "DoubleRainbowLatest",
            f"DoubleRainbow{episodes_per_batch * (batch + 1)}",
        ]:
            plot_best_attempts("./results/", name, "DoubleRainbow", total_rewards)

    return total_losses, total_beta_values, total_td_errors, total_rewards
