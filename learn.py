# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from itertools import count
from tqdm import tqdm
from controls import Controller
from utils import image_to_tensor, select_action, save_results, plot_best_attempts
from rewards import calc_rewards
from DQN import ReplayMemory
from DQN import DQN, optimize_model
import torch.optim as optim
import multiprocessing
from utils import document, load_checkpoint, save_checkpoint
import itertools


def explore_episode(rom_path, timeout, nsteps):
    controller = Controller(rom_path)
    movements = controller.movements
    locs = set()
    xy = set()
    imgs = []
    max_total_level = [0]
    max_total_exp = [0]
    states = []
    stored_states = 0
    for t in tqdm(range(timeout)):
        action = random.randrange(len(movements))
        controller.handleMovement(movements[action])
        img = controller.screen_image()
        reward = calc_rewards(
            controller,
            max_total_level,
            img,
            imgs,
            xy,
            locs,
            max_total_exp,
            default_reward=0.01,
        )

        states.append(controller.create_memory_state(controller))
        if reward > 0.1:
            savname = f"{stored_states}_reward_{reward}_loc_{controller.get_current_location()}_xy_{controller.get_XY()}.state"
            controller.store_state(states[0], savname)
            document(
                0,
                savname,
                img,
                movements[action],
                reward,
                1,
                False,
                timeout,
                1,
                "explore",
            )
            stored_states += 1
        if len(states) > nsteps:
            states.pop(0)


def run_episode(
    i,
    rom_path,
    model,
    epsilon,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    timeout,
    n_steps=100,
    phase=0,
    document_mode=False,
):
    controller = Controller(rom_path)
    movements = controller.movements
    initial_img = controller.screen_image()
    state = image_to_tensor(initial_img, device, SCALE_FACTOR, USE_GRAYSCALE)
    done = False
    n_step_buffer = []
    n_step_buffers = []
    total_reward = 0
    max_total_level = [0]
    max_total_exp = 0
    locs = set(0, 7)
    xy = set()
    imgs = []

    # Initial reward calculation with the first image
    _ = calc_rewards(
        controller,
        max_total_level,
        initial_img,
        imgs,
        xy,
        locs,
        max_total_exp,
        default_reward=0.01,
    )

    for t in count():
        action = select_action(state, epsilon, device, movements, model)
        controller.handleMovement(movements[action.item()])
        img = controller.screen_image()
        reward = calc_rewards(
            controller, max_total_level, img, imgs, xy, locs, default_reward=0.01
        )

        action_tensor = torch.tensor([action], dtype=torch.int64, device=device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
        next_state = (
            image_to_tensor(img, device, SCALE_FACTOR, USE_GRAYSCALE)
            if not done
            else None
        )

        n_step_buffer.append((state, action_tensor, reward_tensor, next_state))

        if len(n_step_buffer) == n_steps or done:
            n_step_buffers.append(
                list(n_step_buffer)
            )  # Shallow copy to avoid deep copy overhead
            n_step_buffer.clear()  # Clear the buffer for new data

        state = next_state
        total_reward += reward
        if done or (timeout > 0 and t >= timeout):
            break
        if document_mode:
            document(
                i,
                t,
                img,
                movements[action.item()],
                reward,
                SCALE_FACTOR,
                USE_GRAYSCALE,
                timeout,
                epsilon,
                phase,
            )

    controller.stop(save=False)

    return n_step_buffers, total_reward


# Function to apply gradients to the model
def apply_gradients(aggregate_gradients, model, optimizer):
    with torch.no_grad():
        for param, grad in zip(model.parameters(), aggregate_gradients):
            param.grad = grad
    optimizer.step()


def log_rewards(batch_rewards):
    return f"Average reward for last batch: {np.mean(batch_rewards)} | Best reward: {np.max(batch_rewards)}"


def chunked_iterable(iterable, size):
    it = iter(iterable)
    for _ in range(0, len(iterable), size):
        yield tuple(itertools.islice(it, size))


def run(
    rom_path,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    timeouts,
    num_episodes,
    episodes_per_batch,
    batch_size,
    nsteps,
    cpus=8,
    explore_mode=False,
):
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
    memory = ReplayMemory(300, n_steps=nsteps, multiCPU=device == torch.device("cpu"))

    start_episode, init_epsilon = load_checkpoint(
        "./checkpoints/", model, optimizer, 0, 1.0
    )
    if explore_mode:
        explore_episode(rom_path, timeouts[0], nsteps)
    else:
        for idx, t in enumerate(timeouts):
            print(f"Timeout: {t}")
            if idx == 0:
                print("Starting Phase 0")
                results = run_phase(
                    init_epsilon,
                    1,
                    0.1,
                    num_episodes,
                    episodes_per_batch,
                    batch_size,
                    t,
                    rom_path,
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
                )
                print("Phase 0 complete\n")

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
                batch_num * episodes_per_batch,
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
        "./results/", batch_num * episodes_per_batch, phase, best_attempts
    )
    return all_rewards


def run_batch(batch_args, cpus):
    if cpus > 1:
        with multiprocessing.Pool(processes=cpus) as pool:
            return pool.starmap(run_episode, batch_args)
    return [run_episode(*args) for args in batch_args]
