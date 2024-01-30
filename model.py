# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import random
from itertools import count
from tqdm import tqdm
import csv
from controls import Controller
from utils import image_to_tensor, select_action, save_results, plot_best_attempts
from rewards import calc_rewards
from DQN import ReplayMemory
from DQN import DQN, optimize_model
import torch.optim as optim
from memory import location as location_address
import multiprocessing
from utils import document, load_checkpoint, save_checkpoint
from time import time
import os
import itertools
import io

def create_memory_state(controller):
    virtual_file = io.BytesIO()
    controller.save_state(virtual_file)
    return virtual_file 

def store_state(state, i):
    # check states folder exists and create if not 
    state.seek(0)
    if not os.path.isdir("./states"):
        os.mkdir("./states")
    with open(f"./states/state_{i}.state", "wb") as f:
        f.write(state.read())

def explore_episode(rom_path, timeout, nsteps):
    controller = Controller(rom_path)
    movements = controller.movements
    locs = set()
    xy = set()
    imgs = []
    rewards = []
    max_total_level = [0]
    max_total_exp = [0]
    states = []
    stored_states = 0
    for t in tqdm(range(timeout)):
        action = random.randrange(len(movements))
        controller.handleMovement(movements[action])
        img = controller.screen_image()
        rewards.append(calc_rewards(controller, max_total_level, img, imgs, xy, locs, max_total_exp, default_reward=0.01))
        states.append(create_memory_state(controller))
        if rewards[-1] > 0.1:
            savname = f"{stored_states}_reward:{rewards[-1]}_loc:{controller.get_current_location()}_xy:{controller.get_XY()}.state"
            store_state(states[0], savname)
            document(0, savname, img, movements[action], rewards[-1], 1, False, timeout, 1, "explore")
            stored_states += 1
        if len(states) > nsteps:
            states.pop(0)

def run_episode(i, rom_path, model, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, n_steps=100, phase=0, document_mode=False):
    controller = Controller(rom_path)
    movements = controller.movements
    state = image_to_tensor(controller.screen_image(), device, SCALE_FACTOR, USE_GRAYSCALE)
    done = False
    all_params = []
    n_step_buffer = []
    n_step_buffers = []
    total_reward = 0
    max_total_level = [0]
    max_total_hp = 0
    max_total_exp = 0
    max_num_pokemon = 0
    max_money = 0
    locs = set()
    xy = set()
    imgs = []
    # this will add the initial xy and locs so that it doesn't trigger as
    # initial reward
    img = controller.screen_image()
    _ = calc_rewards(controller, max_total_level, img, imgs, xy, locs, default_reward=0.01)

    for t in count():
        action = select_action(state, epsilon, device, movements, model)
        controller.handleMovement(movements[action.item()])
        img = controller.screen_image()
        reward = calc_rewards(controller, max_total_level, img, imgs, xy, locs, default_reward=0.01)

        # Convert state and action to tensors
        action_tensor = torch.tensor([action], dtype=torch.int64, device=device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
        next_state = image_to_tensor(img, device, SCALE_FACTOR, USE_GRAYSCALE) if not done else None
        # Append transition to the n_step_buffer
        n_step_buffer.append((state, action_tensor, reward_tensor, next_state))
        # Check if learning should be performed
        if len(n_step_buffer) == n_steps or done:
            n_step_buffers.append(n_step_buffer.copy())
            n_step_buffer = []
        state = next_state
        total_reward += reward
        if done or 0 < timeout <= t:
            break
        if document_mode:
            document(i, t, img, movements[action.item()], reward, SCALE_FACTOR, USE_GRAYSCALE, timeout, epsilon, phase)
    controller.stop(save=False)

    return n_step_buffers, total_reward

# Function to apply gradients to the model
def apply_gradients(aggregate_gradients, model, optimizer):
    with torch.no_grad():
        for param, grad in zip(model.parameters(), aggregate_gradients):
            param.grad = grad
    optimizer.step()


def log_rewards(batch_rewards):
    return (f"Average reward for last batch: {np.mean(batch_rewards)} | Best reward: {np.max(batch_rewards)}")

def chunked_iterable(iterable, size):
    it = iter(iterable)
    for start in range(0, len(iterable), size):
        yield tuple(itertools.islice(it, size))

def run(rom_path, device, SCALE_FACTOR, USE_GRAYSCALE, timeouts, num_episodes, episodes_per_batch, batch_size, nsteps, cpus=8, explore_mode=False):
    controller = Controller(rom_path)
    screen_size = controller.screen_size()
    controller.stop()
    model = DQN(int(screen_size[0] * SCALE_FACTOR), int(screen_size[1] * SCALE_FACTOR), len(controller.movements), USE_GRAYSCALE).to(device)
    if device == torch.device("cpu"):
        model.share_memory()  # Prepare model for shared memory
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = ReplayMemory(300, n_steps=nsteps, multiCPU=True if device == torch.device("cpu") else False)

    # Load checkpoint if it exists
    epsilon_max = 1.0
    epsilon_min = 0.1
    start_episode, init_epsilon = load_checkpoint("./checkpoints/", model, optimizer, 0, epsilon_max)
    episodes_total = 0
    if  explore_mode:
        explore_episode(rom_path, timeouts[0], nsteps)
    else:
        for idx, t in enumerate(timeouts):
            print(f"Timeout: {t}")
            if idx == 0:
                print("Starting Phase 0")
                _ = run_phase(init_epsilon, 1, 1, num_episodes, episodes_per_batch, batch_size, t, rom_path, model, memory, optimizer, device, SCALE_FACTOR, USE_GRAYSCALE, delay_learn=True, checkpoint=True, n_steps=nsteps, phase=f'0_{idx}', cpus=cpus)

                print("Phase 0 complete\n")
            print("Starting Phase 1")

            # RUN PHASE 1
            results = run_phase(init_epsilon, epsilon_max, epsilon_min, num_episodes, episodes_per_batch, batch_size, t, rom_path, model, memory, optimizer, device, SCALE_FACTOR, USE_GRAYSCALE, n_steps=nsteps, delay_learn=True, phase=f'1_{idx}', cpus=cpus)
            print("Phase 1 complete\n")
            print("Starting Phase 2")

            # RUN PHASE 2
            results = run_phase(init_epsilon, epsilon_max/2, epsilon_min, num_episodes, episodes_per_batch, batch_size, t, rom_path, model, memory, optimizer, device, SCALE_FACTOR, USE_GRAYSCALE, n_steps=nsteps, delay_learn=True, phase=f'2_{idx}', cpus=cpus)

            print("Phase 2 complete\n")
            print("Starting Phase 3")

            # RUN PHASE 3
            results = run_phase(init_epsilon, epsilon_max/4, epsilon_min, num_episodes, episodes_per_batch, batch_size, t, rom_path, model, memory, optimizer, device, SCALE_FACTOR, USE_GRAYSCALE, n_steps=nsteps, delay_learn=True, phase=f'3_{idx}', cpus=cpus)

            print("Phase 3 complete\n")
            print("Done...")


        # Save final model
        save_checkpoint("./checkpoints/", model, optimizer, episodes_total, epsilon_min, timeouts[-1])

        # Save results
        save_results("./results/", 1, results)

def eval_model(rom_path, model, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, nsteps, batch_num, phase):
    reward = run_episode(batch_num, rom_path, model, 0, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, n_steps=nsteps, document_mode=True, phase=phase)
    return reward

def run_phase(init_epsilon, epsilon_max, epsilon_min, num_episodes, episodes_per_batch, batch_size, timeout, rom_path, model, memory, optimizer, device, SCALE_FACTOR, USE_GRAYSCALE, n_steps=100, delay_learn=False, checkpoint=True, phase=0, cpus=8):
    all_rewards = []
    # Main loop
    epsilon = init_epsilon
    decay_rate = -np.log(epsilon_min / epsilon_max) / num_episodes
    adjusted_decay_rate = decay_rate / 2  # or any other factor > 1 to slow down the decay
    epsilons_exponential = epsilon_max * np.exp(-adjusted_decay_rate * np.arange(num_episodes))

    args = [(i, rom_path, model, epsilons_exponential[i], device, SCALE_FACTOR, USE_GRAYSCALE, timeout, n_steps, phase) for i in range(num_episodes)]
    batch_vals = []
    best_attempts = []
    # Split the episodes into batches and run them in parallel
    for batch_num, batch_args in enumerate(tqdm(chunked_iterable(args, episodes_per_batch),
                                                total=len(args) // episodes_per_batch, desc="Awaiting results...")):
        
        try:
            if cpus > 1:
                with multiprocessing.Pool(processes=cpus) as pool:
                    batch_results = pool.starmap(run_episode, batch_args)
            else:
                batch_results = [run_episode(*args) for args in batch_args]
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            continue

        for run_i, batch_rewards in batch_results:
            all_rewards.append(batch_rewards)
            batch_vals.append(batch_rewards)
            for sequences in run_i:
                for sequence in sequences:
                    memory.push(*sequence)

        if len(memory) <= batch_size:
            optimize_model(len(memory), device, memory, model, optimizer, n_steps=n_steps)  
        else:
            optimize_model(batch_size, device, memory, model, optimizer, n_steps=n_steps)  

        if checkpoint and batch_num % 100 == 0:
            save_checkpoint("./checkpoints/", model, optimizer, batch_num * episodes_per_batch, epsilon, timeout)
        _, best_attempt = eval_model(rom_path, model, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, n_steps, batch_num, phase)
        best_attempts.append(best_attempt)

    plot_best_attempts("./results/", batch_num * episodes_per_batch, phase, best_attempts)
    return all_rewards
