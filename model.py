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


def run_episode(rom_path, model, memory, optimizer, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size):
    controller = Controller(rom_path)
    movements = controller.movements
    state = image_to_tensor(controller.screen_image(), device, SCALE_FACTOR, USE_GRAYSCALE)
    visited_locations = set()
    total_reward = 0
    max_levels = [0]
    positive_keywords = {"received": False, "won": False}
    negative_keywords = {
        "lost": False,
        "fainted": False,
        "save the game": False,
        "saving": False,
        "select": False,
        "town map": False,
        "text speed": False,
    }

    episode_gradients = []

    for t in count():
        action = select_action(state, epsilon, device, movements, model)
        controller.handleMovement(movements[action.item()])
        reward = torch.tensor([-0.01], dtype=torch.float32, device=device)
        img = controller.screen_image()
        loc = controller.get_memory_value(location_address)
        done = False
        reward = calc_rewards(
            movements[action.item()], loc, visited_locations, controller, positive_keywords, negative_keywords, max_levels, default_reward=0.01
        )

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

        if done or 0 < timeout < t:
            break

    controller.stop(save=False)
    return episode_gradients

# Function to apply gradients to the model
def apply_gradients(aggregate_gradients, model, optimizer):
    with torch.no_grad():
        for param, grad in zip(model.parameters(), aggregate_gradients):
            param.grad = grad
    optimizer.step()

# Run episodes in parallel and collect gradients
def run_episodes_batch(start, end, rom_path, model, memory, optimizer, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size):
    batch_gradients = []
    for i in range(start, end):
        episode_gradients = run_episode(rom_path, model, memory, optimizer, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size)
        batch_gradients.append(episode_gradients)
    return batch_gradients

def run(rom_path, device, SCALE_FACTOR, USE_GRAYSCALE,  timeout, num_episodes=100, episodes_per_batch=5, batch_size=128, epsilon=1.0 ):

    controller = Controller(rom_path)
    screen_size = controller.screen_size()
    controller.stop()
    model = DQN(int(screen_size[0] * SCALE_FACTOR), int(screen_size[1] * SCALE_FACTOR), len(controller.movements), USE_GRAYSCALE).to(device)
    model.share_memory()  # Prepare model for shared memory
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    memory = ReplayMemory(1000)



    # Main loop
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Create a list of arguments for each batch
        batch_args = [(i, min(i + episodes_per_batch, num_episodes), rom_path, model, memory, optimizer, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size)
                      for i in range(0, num_episodes, episodes_per_batch)]

        # Process the batches in parallel
        all_batch_gradients = pool.starmap(run_episodes_batch, batch_args)

        for batch_gradients in tqdm(all_batch_gradients):
            # Aggregate gradients
            aggregate_gradients = [sum(grads) / len(grads) for grads in zip(*batch_gradients)]

            # Update the model
            apply_gradients(aggregate_gradients, model, optimizer)

    # Save final model
    torch.save(model.state_dict(), "./checkpoints/pokemon_rl_model_final.pth")