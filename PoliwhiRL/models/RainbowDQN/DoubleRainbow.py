# -*- coding: utf-8 -*-
from multiprocessing import Pool
import torch
import random
from tqdm import tqdm
from .utils import (
    compute_td_error,
    optimize_model,
    beta_by_frame,
    save_checkpoint,
    epsilon_by_frame_cyclic,
)
from PoliwhiRL.utils.utils import image_to_tensor
from PoliwhiRL.environment import Controller


def worker(
    worker_id,
    rom_path,
    state_path,
    episode_length,
    sight,
    frame_start,
    epsilon_start,
    epsilon_final,
    epsilon_decay,
    beta_start,
    beta_frames,
    policy_net,
    target_net,
    device,
    num_episodes,
    frames_in_loc,
    epsilon_by_location,
    extra_files,
    reward_locations_xy,
    scaling_factor,
):
    local_env = Controller(
        rom_path,
        state_path,
        timeout=episode_length,
        log_path=f"./logs/double_rainbow_env_{worker_id}.json",
        use_sight=sight,
        extra_files=extra_files,
        reward_locations_xy=reward_locations_xy,
        scaling_factor=scaling_factor,
    )
    experiences, rewards, td_errors, frame_idxs, epsilon_values = [], [], [], [], []

    frame_idx = frame_start
    for episode in range(num_episodes):
        state = local_env.reset()
        state = image_to_tensor(state, device)
        total_reward = 0
        episode_td_errors = []
        episode_experiences = []
        done = False
        while not done:
            frame_idx += 1
            frames_in_loc[local_env.get_current_location()] += 1
            frame_idxs.append(frame_idx)
            epsilon = epsilon_by_frame_cyclic(
                (
                    frames_in_loc[local_env.get_current_location()]
                    if epsilon_by_location
                    else frame_idx
                ),
                epsilon_start,
                epsilon_final,
                epsilon_decay,
            )
            epsilon_values.append(epsilon)
            beta = beta_by_frame(
                (
                    frames_in_loc[local_env.get_current_location()]
                    if epsilon_by_location
                    else frame_idx
                ),
                beta_start,
                beta_frames,
            )
            if random.random() > epsilon:
                with torch.no_grad():
                    state_t = state.unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    action = q_values.max(1)[1].item()
            else:
                action = local_env.random_move()

            next_state, reward, done = local_env.step(action)
            next_state = image_to_tensor(next_state, device)
            total_reward += reward

            action = torch.tensor([action], device=device)
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)
            td_error = compute_td_error(
                (state, action, reward, next_state, done),
                policy_net,
                target_net,
                device,
            )
            episode_experiences.append(
                (state, action, reward, next_state, done, beta, td_error)
            )
            state = next_state
        experiences.extend(episode_experiences)
        rewards.append(total_reward)
        td_errors.extend(episode_td_errors)

    local_env.close()
    return experiences, rewards, td_errors, frame_idxs, epsilon_values


def aggregate_and_update_model(
    experiences,
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    device,
    batch_size,
    gamma,
    update_target_every,
    memories_processed,
):
    losses = []
    for experience in experiences:
        state, action, reward, next_state, done, beta, td_error = experience
        replay_buffer.add(state, action, reward, next_state, done, error=td_error)
        memories_processed += 1
        # Check if it's time to update the target network
        if memories_processed % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())
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
    return losses, memories_processed


def run(
    rom_path,
    state_path,
    episode_length,
    device,
    num_episodes,
    batch_size,
    sight,
    runs_per_worker,
    num_workers,
    memories,
    epsilon_start,
    epsilon_final,
    epsilon_decay,
    beta_start,
    beta_frames,
    gamma,
    update_target_every,
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    losses,
    rewards,
    checkpoint_interval,
    checkpoint_path,
    epsilon_by_location,
    frames_in_loc,
    extra_files,
    reward_locations_xy,
    scaling_factor,
):
    batches_to_run = num_episodes // (num_workers * runs_per_worker)
    if batches_to_run == 0:
        raise ValueError(
            "Not enough episodes to run the model. Increase num_episodes or decrease num_workers and runs_per_worker."
        )
    for run in tqdm(range(batches_to_run), desc="Running..."):
        new_results, new_losses, new_memories = run_batch(
            run,
            num_workers,
            rom_path,
            state_path,
            episode_length,
            memories,
            device,
            runs_per_worker,
            batch_size,
            policy_net,
            target_net,
            optimizer,
            replay_buffer,
            sight,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            beta_start,
            beta_frames,
            gamma,
            update_target_every,
            losses,
            frames_in_loc,
            epsilon_by_location,
            extra_files,
            reward_locations_xy,
            scaling_factor,
        )
        memories += new_memories
        rewards.extend(new_results)
        losses.extend(new_losses)

        if run % checkpoint_interval == 0 and run > 0:
            save_checkpoint(
                {
                    "episode": (run + 1) * num_workers * runs_per_worker,
                    "frame_idx": memories,
                    "policy_net_state_dict": policy_net.state_dict(),
                    "target_net_state_dict": target_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "replay_buffer": replay_buffer.state_dict(),
                    "frames_in_loc": frames_in_loc,
                },
                filename=checkpoint_path,
            )

    return losses, rewards, memories


def run_batch(
    batch_n,
    num_workers,
    rom_path,
    state_path,
    episode_length,
    memories,
    device,
    num_episodes,
    batch_size,
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    sight,
    epsilon_start,
    epsilon_final,
    epsilon_decay,
    beta_start,
    beta_frames,
    gamma,
    update_target_every,
    losses,
    frames_in_loc,
    epsilon_by_location,
    extra_files,
    reward_locations_xy,
    scaling_factor,
):
    # Prepare arguments for each worker function call
    args_list = [
        (
            num_workers * batch_n + i,
            rom_path,
            state_path,
            episode_length,
            sight,
            memories,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            beta_start,
            beta_frames,
            policy_net,
            target_net,
            device,
            num_episodes,
            frames_in_loc,
            epsilon_by_location,
            extra_files,
            reward_locations_xy,
            scaling_factor,
        )
        for i in range(num_workers)
    ]

    # Initialize a multiprocessing pool
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(worker, args_list)

    # Process the results returned from the workers
    experiences, rewards, td_errors, frame_idxs, epsilon_values = [], [], [], [], []
    for result in results:
        (
            worker_experiences,
            worker_rewards,
            worker_td_errors,
            worker_frame_idxs,
            worker_epsilon_values,
        ) = result
        experiences.extend(worker_experiences)
        rewards.extend(worker_rewards)
        td_errors.extend(worker_td_errors)
        frame_idxs.extend(worker_frame_idxs)
        epsilon_values.extend(worker_epsilon_values)

    memories_processed = memories + len(experiences)
    if len(experiences) > 0:
        loss, memories_processed = aggregate_and_update_model(
            experiences,
            policy_net,
            target_net,
            optimizer,
            replay_buffer,
            device,
            batch_size,
            gamma,
            update_target_every,
            memories_processed,
        )
        if loss is not None:
            losses.extend(
                loss
            )  # Assuming aggregate_and_update_model returns a list of losses

    return rewards, losses, memories_processed
