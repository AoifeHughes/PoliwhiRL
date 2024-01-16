# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import random
from itertools import count
from tqdm import tqdm
import csv
from controls import Controller
from utils import image_to_tensor, select_action
from rewards import calc_rewards
from TorchModel import ReplayMemory
from TorchModel import DQN
import torch.optim as optim
from memory import location as location_address
import multiprocessing
from utils import document, load_checkpoint, save_checkpoint
from time import time
import os
import itertools

def optimize_model(batch_size, device, memory, model, optimizer):
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    batch = tuple(zip(*transitions))

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(batch_size, device=device)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch[3])),
        dtype=torch.bool,
        device=device,
    )
    non_final_next_states = torch.cat([s for s in batch[3] if s is not None])
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * 0.99) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backward pass to compute gradients
    optimizer.zero_grad()
    loss.backward()

    # Extract and return gradients
    gradients = [param.grad.clone() for param in model.parameters() if param.grad is not None]
    return gradients


def run_episode(episode_id, rom_path, model, memory, optimizer, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size):
    controller = Controller(rom_path)
    movements = controller.movements
    state = image_to_tensor(controller.screen_image(), device, SCALE_FACTOR, USE_GRAYSCALE)
    total_reward = 0
    max_levels = [0]

    episode_gradients = []
    total_reward = 0
    imgs = []
    for t in count():
        img_orig = controller.screen_image()
        action = select_action(state, epsilon, device, movements, model)
        controller.handleMovement(movements[action.item()])
        img = controller.screen_image()
        done = False
        reward = calc_rewards(controller, max_levels, img, imgs, default_reward=0.01
        ) 
        total_reward += reward
        next_state = image_to_tensor(img, device, SCALE_FACTOR, USE_GRAYSCALE) if not done else None

        memory.push(
            state,
            action,
            torch.tensor([reward], dtype=torch.float32, device=device),
            next_state,
        )

        # Call optimize_model but modify it to return gradients
        gradients = optimize_model(batch_size, device, memory, model, optimizer)
        if gradients:
            episode_gradients.append(gradients)

        total_reward += reward
        state = next_state
        document(episode_id, t, img_orig, movements[action.item()], reward, SCALE_FACTOR, USE_GRAYSCALE, timeout, epsilon)
        if done or 0 < timeout < t:
            break

    controller.stop(save=False)
    return episode_gradients, total_reward

# Function to apply gradients to the model
def apply_gradients(aggregate_gradients, model, optimizer):
    with torch.no_grad():
        for param, grad in zip(model.parameters(), aggregate_gradients):
            param.grad = grad
    optimizer.step()

# Run episodes in parallel and collect gradients
def run_episodes_batch(i, reps, rom_path, model, memory, optimizer, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size):
    batch_gradients = []
    batch_rewards = []
    for j in range(i, reps+i):
        episode_gradients, episode_rewards = run_episode(j, rom_path, model, memory, optimizer, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size)
        batch_gradients.append(episode_gradients)
        batch_rewards.append(episode_rewards)
    return batch_gradients, batch_rewards

def log_rewards(rewards):
    print(f"Average reward for batch: {np.mean(rewards)}")

def chunked_iterable(iterable, size):
    it = iter(iterable)
    for start in range(0, len(iterable), size):
        yield tuple(itertools.islice(it, size))

def run(rom_path, device, SCALE_FACTOR, USE_GRAYSCALE, timeouts, num_episodes=100, episodes_per_batch=5, batch_size=128):

    controller = Controller(rom_path)
    screen_size = controller.screen_size()
    controller.stop()
    model = DQN(int(screen_size[0] * SCALE_FACTOR), int(screen_size[1] * SCALE_FACTOR), len(controller.movements), USE_GRAYSCALE).to(device)
    model.share_memory()  # Prepare model for shared memory
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = ReplayMemory(1000)
    # Load checkpoint if it exists
    epsilon_max = 1.0
    epsilon_min = 0.1
    start_episode, init_epsilon = load_checkpoint("./checkpoints/", model, optimizer, 0, epsilon_max)
    all_rewards = {}
    start_time = time()
    episodes_total = 0
    reps = os.cpu_count()
    for timeout in timeouts:
        all_rewards[timeout] = []
        # Main loop
        epsilon = init_epsilon
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            # calculate an array of epsilons for each episode which decay
            # exponentially from epsilon_max to epsilon_min
            decay_rate = -np.log(epsilon_min / epsilon_max) / num_episodes*reps

            # Calculate the epsilon values for each episode
            epsilons_exponential = epsilon_max * np.exp(-decay_rate * np.arange(num_episodes*reps))
            # Create a list of arguments for each batch
            args = [(i, reps,
                            rom_path, model, memory, optimizer, 
                            epsilons_exponential[i], device, SCALE_FACTOR,
                              USE_GRAYSCALE, timeout, batch_size)
                            for i in range(0, num_episodes*reps, reps)]
            
            # split the episodes into batches and run them in parallel
            for batch_args in tqdm(chunked_iterable(args, episodes_per_batch), total=num_episodes//episodes_per_batch):
                for batch_results in pool.starmap(run_episodes_batch, batch_args):
                    batch_gradients, batch_rewards = batch_results
                    # remove empty gradients
                    batch_gradients = [grad for grad in batch_gradients if len(grad) > 0]
                    if len(batch_gradients) == 0:
                        pass
                    # Aggregate gradients
                    # check contents of batch_gradients
                    aggregate_gradients = [torch.mean(torch.stack(grads), dim=0) for grads in zip(*batch_gradients) if all(isinstance(g, torch.Tensor) for g in grads)]
                    # Update the model
                    apply_gradients(aggregate_gradients, model, optimizer)
                    all_rewards[timeout].append(batch_rewards)
                log_rewards(batch_rewards)

        episodes_total += len(all_rewards[timeout])
        save_checkpoint("./checkpoints/", model, optimizer, episodes_total, epsilon, timeout)

        # save data with number of episodes completed and timeout
        with open(f"./checkpoints/all_rewards_timeout_{timeout}_episodetotal_{episodes_per_batch*episodes_total+start_episode}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(all_rewards[timeout])
        
        print("Time elapsed: ", time() - start_time)
            
    # Save final model
    save_checkpoint("./checkpoints/", model, optimizer, episodes_total, epsilon, timeout)

