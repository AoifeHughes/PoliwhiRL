# -*- coding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm
from controls import Controller
from utils import (
    save_results,
    plot_best_attempts,
    load_checkpoint,
    save_checkpoint,
)
from episodes import run_episode, explore_episode
from DQN import (
    DQN,
    ReplayMemory,
)  # Ensure optimize_model is correctly imported or defined
import torch.optim as optim
import multiprocessing


def run(
    rom_path,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    timeouts,
    state_paths,
    num_episodes,
    episodes_per_batch,
    batch_size,
    nsteps,
    cpus=8,
    explore_mode=False,
):
    if explore_mode:
        explore_episode(rom_path, timeouts[0], nsteps)
    else:
        screen_size = Controller(rom_path).screen_size()
        # Initialize primary model
        primary_model = DQN(
            int(screen_size[0] * SCALE_FACTOR),
            int(screen_size[1] * SCALE_FACTOR),
            len(Controller(rom_path).movements),
            USE_GRAYSCALE,
        ).to(device)
        # Initialize target model and copy weights from the primary model
        target_model = DQN(
            int(screen_size[0] * SCALE_FACTOR),
            int(screen_size[1] * SCALE_FACTOR),
            len(Controller(rom_path).movements),
            USE_GRAYSCALE,
        ).to(device)
        target_model.load_state_dict(primary_model.state_dict())
        target_model.eval()  # Set target model to evaluation mode

        if device == torch.device("cpu"):
            primary_model.share_memory()  # Prepare model for shared memory
        optimizer = optim.Adam(primary_model.parameters(), lr=0.001)
        memory = ReplayMemory(
            10000, n_steps=nsteps, multiCPU=device == torch.device("cpu")
        )
        epsilon = 1
        start_episode, init_epsilon = load_checkpoint(
            "./checkpoints/", primary_model, optimizer, 0, 1.0
        )

        for idy, t in enumerate(timeouts):
            print(f"Timeout: {t}")
            for idx, state_path in enumerate(state_paths):
                print(f"Starting Phase {idx}")
                results = run_phase(
                    init_epsilon,
                    epsilon - idy * 0.1,
                    epsilon - idy * 0.1,
                    num_episodes,
                    episodes_per_batch,
                    batch_size,
                    t,
                    rom_path,
                    state_path,
                    primary_model,
                    target_model,  # Pass target model to run_phase
                    memory,
                    optimizer,
                    device,
                    SCALE_FACTOR,
                    USE_GRAYSCALE,
                    nsteps,
                    checkpoint=True,
                    phase=f"{t}_{idx}",
                    cpus=cpus,
                    start_episode=start_episode,
                )
                print(f"Phase {idx} complete\n")
                save_checkpoint(
                    "./checkpoints/",
                    primary_model,
                    optimizer,
                    num_episodes,
                    0.1,
                    timeouts[-1],
                )
                start_episode += num_episodes
                save_results("./results/", start_episode, results)


def eval_model(
    rom_path,
    state_path,
    model,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    timeout,
    nsteps,
    batch_num,
    phase,
):
    reward = run_episode(
        batch_num,
        rom_path,
        state_path,
        model,
        0,
        device,
        SCALE_FACTOR,
        USE_GRAYSCALE,
        timeout,
        n_steps=nsteps,
        document_mode=True,
        phase=phase,
    )
    return reward


def run_phase(
    init_epsilon,
    epsilon_max,
    epsilon_min,
    num_episodes,
    episodes_per_batch,
    batch_size,
    timeout,
    rom_path,
    state_path,
    primary_model,
    target_model,
    memory,
    optimizer,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    n_steps=100,
    checkpoint=True,
    phase=0,
    cpus=8,
    start_episode=0,
):
    all_rewards = []
    epsilons_exponential = np.linspace(epsilon_max, epsilon_min, num_episodes)

    for batch_start in tqdm(
        range(0, num_episodes, episodes_per_batch), desc="Running batches"
    ):
        epsilons_batch = epsilons_exponential[
            batch_start : batch_start + episodes_per_batch
        ]
        batch_args = [
            (
                i + batch_start,
                rom_path,
                state_path,
                primary_model,
                target_model,
                epsilons_batch[i - batch_start],
                device,
                memory,
                optimizer,
                SCALE_FACTOR,
                USE_GRAYSCALE,
                timeout,
                n_steps,
                batch_size,
                phase,
                False,  # Assuming document_mode is False by default
            )
            for i in range(
                batch_start, min(batch_start + episodes_per_batch, num_episodes)
            )
        ]

        batch_results = run_batch(batch_args, cpus)

        # Aggregate rewards and possibly other metrics from batch_results
        for total_reward in batch_results:
            all_rewards.append(total_reward)

        # Save checkpoint periodically or based on other criteria
        if checkpoint and (batch_start + episodes_per_batch) % 100 == 0:
            save_checkpoint(
                "./checkpoints/",
                primary_model,
                optimizer,
                batch_start + episodes_per_batch,
                epsilons_exponential[
                    min(batch_start + episodes_per_batch, num_episodes - 1)
                ],
                timeout,
            )

    # Optional: plot best attempts or other metrics collected during the phase
    plot_best_attempts(
        "./results/",
        start_episode + num_episodes,
        phase,
        all_rewards,
    )

    return all_rewards


def run_batch(batch_args, cpus):
    if cpus > 1:
        with multiprocessing.Pool(processes=cpus) as pool:
            return pool.starmap(run_episode, batch_args)
    else:
        return [run_episode(*args) for args in batch_args]
