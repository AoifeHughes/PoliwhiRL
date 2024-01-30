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
    chunked_iterable,
)
from episodes import run_episode, explore_episode
from DQN import DQN, optimize_model, ReplayMemory
import torch.optim as optim
import multiprocessing


# Function to apply gradients to the model
def apply_gradients(aggregate_gradients, model, optimizer):
    with torch.no_grad():
        for param, grad in zip(model.parameters(), aggregate_gradients):
            param.grad = grad
    optimizer.step()


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
        model = DQN(
            int(screen_size[0] * SCALE_FACTOR),
            int(screen_size[1] * SCALE_FACTOR),
            len(Controller(rom_path).movements),
            USE_GRAYSCALE,
        ).to(device)
        if device == torch.device("cpu"):
            model.share_memory()  # Prepare model for shared memory
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        memory = ReplayMemory(
            300, n_steps=nsteps, multiCPU=device == torch.device("cpu")
        )

        start_episode, init_epsilon = load_checkpoint(
            "./checkpoints/", model, optimizer, 0, 1.0
        )
        for _, t in enumerate(timeouts):
            print(f"Timeout: {t}")
            for idx, state_path in enumerate(state_paths):
                print(f"Starting Phase {idx}")
                results = run_phase(
                    init_epsilon,
                    1,
                    0.1,
                    num_episodes,
                    episodes_per_batch,
                    batch_size,
                    t,
                    rom_path,
                    state_path,
                    model,
                    memory,
                    optimizer,
                    device,
                    SCALE_FACTOR,
                    USE_GRAYSCALE,
                    nsteps,
                    delay_learn=True,
                    checkpoint=True,
                    phase=f"0_{idx}",
                    cpus=cpus,
                    start_episode=start_episode,
                )
                print(f"Phase {idx} complete\n")

        save_checkpoint(
            "./checkpoints/", model, optimizer, num_episodes, 0.1, timeouts[-1]
        )
        save_results("./results/", 1, results)


def eval_model(
    rom_path,
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
    model,
    memory,
    optimizer,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    n_steps=100,
    delay_learn=False,
    checkpoint=True,
    phase=0,
    cpus=8,
    start_episode=0,
):
    all_rewards = []
    epsilon = init_epsilon
    decay_rate = -np.log(epsilon_min / epsilon_max) / num_episodes
    adjusted_decay_rate = decay_rate / 2
    epsilons_exponential = epsilon_max * np.exp(
        -adjusted_decay_rate * np.arange(num_episodes)
    )

    args = [
        (
            i,
            rom_path,
            state_path,
            model,
            epsilons_exponential[i],
            device,
            SCALE_FACTOR,
            USE_GRAYSCALE,
            timeout,
            n_steps,
            phase,
        )
        for i in range(num_episodes)
    ]
    batch_vals = []
    best_attempts = []

    for batch_num, batch_args in enumerate(
        tqdm(
            chunked_iterable(args, episodes_per_batch),
            total=len(args) // episodes_per_batch,
            desc="Awaiting results...",
        )
    ):
        batch_results = run_batch(batch_args, cpus)

        for run_i, batch_rewards in batch_results:
            all_rewards.append(batch_rewards)
            batch_vals.append(batch_rewards)
            for sequences in run_i:
                for sequence in sequences:
                    memory.push(*sequence)

        optimize_model(
            min(len(memory), batch_size),
            device,
            memory,
            model,
            optimizer,
            n_steps=n_steps,
        )

        if checkpoint and batch_num % 100 == 0:
            save_checkpoint(
                "./checkpoints/",
                model,
                optimizer,
                batch_num * episodes_per_batch + start_episode,
                epsilon,
                timeout,
            )
        _, best_attempt = eval_model(
            rom_path,
            model,
            device,
            SCALE_FACTOR,
            USE_GRAYSCALE,
            timeout,
            n_steps,
            batch_num,
            phase,
        )
        best_attempts.append(best_attempt)

    plot_best_attempts(
        "./results/",
        batch_num * episodes_per_batch + start_episode,
        phase,
        best_attempts,
    )
    return all_rewards


def run_batch(batch_args, cpus):
    if cpus > 1:
        with multiprocessing.Pool(processes=cpus) as pool:
            return pool.starmap(run_episode, batch_args)
    return [run_episode(*args) for args in batch_args]
