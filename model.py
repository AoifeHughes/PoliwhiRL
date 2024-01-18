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

def optimize_model(batch_size, device, memory, model, optimizer, gamma=0.99, n_steps=5):
    if len(memory) < batch_size:
        return None

    sequences = memory.sample(batch_size)

    state_batch, action_batch, reward_batch, next_state_batch, non_final_mask = [], [], [], [], []

    for sequence in sequences:
        # Unzip the sequence
        states, actions, rewards, next_states = zip(*sequence)

        # Process rewards
        rewards = [r[0].item() for r in rewards]

        # Calculate the cumulative reward for the N steps
        cumulative_reward = sum([gamma**i * rewards[i] for i in range(len(rewards))])

        # Process states, actions, and next_states
        processed_state = states[0][0] if isinstance(states[0], tuple) else states[0]
        processed_action = actions[0][0] if isinstance(actions[0], tuple) else actions[0]
        processed_next_state = next_states[-1][0] if isinstance(next_states[-1], tuple) and next_states[-1] is not None else next_states[-1]

        state_batch.append(processed_state)
        action_batch.append(processed_action)
        reward_batch.append(cumulative_reward)
        next_state_batch.append(processed_next_state)
        non_final_mask.append(processed_next_state is not None)

    # Convert batches to tensors
    state_batch = torch.cat(state_batch, dim=0)
    action_batch = torch.cat(action_batch, dim=0)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
    non_final_next_states = torch.cat([ns for ns in next_state_batch if ns is not None], dim=0)



    # Compute Q(s_t, a)
    state_action_values = model(state_batch).gather(1, action_batch.unsqueeze(1))

    # Compute V(s_{t+N}) for all non-final next states
    next_state_values = torch.zeros(batch_size, device=device)
    non_final_mask = torch.tensor(non_final_mask, dtype=torch.bool, device=device)
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * (gamma ** n_steps)) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Extract gradients
    gradients = [param.grad.clone() for param in model.parameters() if param.grad is not None]
    return gradients


def learn(batch_size, device, memory, model, optimizer, episode_gradients, n_step_buffer):
    # Push the N-step sequence to memory
    memory.push(*zip(*n_step_buffer))
    
    # Call optimize_model to update the model with batches of N-step sequences
    gradients = optimize_model(batch_size, device, memory, model, optimizer, n_steps=len(n_step_buffer))
    
    # Store the gradients for the episode
    if gradients:
        episode_gradients.append(gradients)


def run_episode(episode_id, rom_path, model, memory, optimizer, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size, n_steps=100, delay_learn=False):
    controller = Controller(rom_path)
    movements = controller.movements
    state = image_to_tensor(controller.screen_image(), device, SCALE_FACTOR, USE_GRAYSCALE)
    done = False
    episode_gradients = []
    all_params = []
    n_step_buffer = []

    total_reward = 0
    max_total_level = [0]
    max_total_hp = 0
    max_total_exp = 0
    max_num_pokemon = 0
    max_money = 0
    locs = set()
    xy = set()
    imgs = []

    controller.handleMovement(movements[0])

    for t in count():
        action = select_action(state, epsilon, device, movements, model)
        controller.handleMovement(movements[action.item()])

        img = controller.screen_image()
        #done = controller.is_done()  # Updated to check the terminal state from the controller
        reward = calc_rewards(controller, max_total_level, img, imgs, xy, locs, default_reward=0.01)
        
        # Convert state and action to tensors
        action_tensor = torch.tensor([action], dtype=torch.int64, device=device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
        next_state = image_to_tensor(img, device, SCALE_FACTOR, USE_GRAYSCALE) if not done else None

        # Append transition to the n_step_buffer
        n_step_buffer.append((state, action_tensor, reward_tensor, next_state))

        # Check if learning should be performed
        if len(n_step_buffer) == n_steps or done:
            learn_data = (batch_size, device, memory, model, optimizer, episode_gradients, n_step_buffer.copy())
            if delay_learn:
                all_params.append(learn_data)
            else:
                learn(*learn_data)
            n_step_buffer = []

        state = next_state
        total_reward += reward
        document(episode_id, t, controller.screen_image(), movements[action.item()], reward, SCALE_FACTOR, USE_GRAYSCALE, timeout, epsilon)

        if done or 0 < timeout <= t:
            break

    # Perform delayed learning if specified
    if delay_learn:
        for params in tqdm(all_params, desc="Learning..."):
            learn(*params)

    controller.stop(save=False)
    return episode_gradients, total_reward

# Function to apply gradients to the model
def apply_gradients(aggregate_gradients, model, optimizer):
    with torch.no_grad():
        for param, grad in zip(model.parameters(), aggregate_gradients):
            param.grad = grad
    optimizer.step()


def log_rewards(batch_rewards):
    return (f"Average reward for last batch: {np.mean(batch_rewards[-1])} | Best reward average: {np.max(np.mean(batch_rewards))}")

def chunked_iterable(iterable, size):
    it = iter(iterable)
    for start in range(0, len(iterable), size):
        yield tuple(itertools.islice(it, size))

def run(rom_path, device, SCALE_FACTOR, USE_GRAYSCALE, timeouts, num_episodes=100, episodes_per_batch=os.cpu_count(), batch_size=32):
    controller = Controller(rom_path)
    screen_size = controller.screen_size()
    controller.stop()
    model = DQN(int(screen_size[0] * SCALE_FACTOR), int(screen_size[1] * SCALE_FACTOR), len(controller.movements), USE_GRAYSCALE).to(device)
    model.share_memory()  # Prepare model for shared memory
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = ReplayMemory(100000)

    # Load checkpoint if it exists
    epsilon_max = 1.0
    epsilon_min = 0.1
    start_episode, init_epsilon = load_checkpoint("./checkpoints/", model, optimizer, 0, epsilon_max)
    episodes_total = 0

    # RUN PHASE 0 - not in parallel to avoid mem problems
    print("Starting Phase 0")
    _ = run_phase(init_epsilon, 1, 1, 1, 1, batch_size, 2000, rom_path, model, memory, optimizer, device, SCALE_FACTOR, USE_GRAYSCALE, delay_learn=True, checkpoint=False, n_steps=5)
    print("Phase 0 complete\n")
    print("Starting Phase 1")

    # RUN PHASE 1
    results = run_phase(init_epsilon, epsilon_max, epsilon_min, num_episodes, episodes_per_batch, batch_size, timeouts[0], rom_path, model, memory, optimizer, device, SCALE_FACTOR, USE_GRAYSCALE)

    print("Phase 1 complete\n")
    print("Starting Phase 2")

    # RUN PHASE 2
    results = run_phase(init_epsilon, epsilon_max/2, epsilon_min, num_episodes, episodes_per_batch, batch_size, timeouts[0], rom_path, model, memory, optimizer, device, SCALE_FACTOR, USE_GRAYSCALE)

    print("Phase 2 complete\n")

    # Save final model
    save_checkpoint("./checkpoints/", model, optimizer, episodes_total, epsilon_min, timeouts[-1])

    # Save results to file
    with open(f"results_{time()}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)



def run_phase(init_epsilon, epsilon_max, epsilon_min, num_episodes, episodes_per_batch, batch_size, timeout, rom_path, model, memory, optimizer, device, SCALE_FACTOR, USE_GRAYSCALE, n_steps=100, delay_learn=False, checkpoint=True):
    all_rewards= []
    # Main loop
    epsilon = init_epsilon
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        decay_rate = -np.log(epsilon_min / epsilon_max) / num_episodes
        adjusted_decay_rate = decay_rate / 2  # or any other factor > 1 to slow down the decay
        epsilons_exponential = epsilon_max * np.exp(-adjusted_decay_rate * np.arange(num_episodes))

        args = [(i, rom_path, model, memory, optimizer, epsilons_exponential[i], device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size, n_steps, delay_learn) for i in range(num_episodes)]
        
        # Split the episodes into batches and run them in parallel
        for batch_num, batch_args in enumerate(tqdm(chunked_iterable(args, episodes_per_batch),
                                                        total=len(args)//episodes_per_batch, desc="Awaiting results...")):
            batch_results = pool.starmap(run_episode, batch_args)
            for batch_gradients, batch_rewards in batch_results:
                batch_gradients = [grad for grad in batch_gradients if len(grad) > 0]
                if len(batch_gradients) == 0:
                    pass
                aggregate_gradients = [torch.mean(torch.stack(grads), dim=0) for grads in zip(*batch_gradients) if all(isinstance(g, torch.Tensor) for g in grads)]
                # Update the model
                apply_gradients(aggregate_gradients, model, optimizer)
                all_rewards.append(batch_rewards)

            if checkpoint:
                save_checkpoint("./checkpoints/", model, optimizer, batch_num*episodes_per_batch, epsilon, timeout)

    return all_rewards
