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

def optimize_model( batch_size, device, memory, model, optimizer):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = tuple(zip(*transitions))

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(
        batch_size, device=device
    )  # Tensor initialized on the specified device
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch[3])),
        dtype=torch.bool,
        device=device,
    )  # Tensor initialized on the specified device

    non_final_next_states = torch.cat([s for s in batch[3] if s is not None])
    next_state_values[non_final_mask] = (
        model(non_final_next_states).max(1)[0].detach()
    )

    expected_state_action_values = (next_state_values * 0.99) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def run_episode(rom_path, model, memory, optimizer, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size ):
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

        optimize_model(batch_size, device, memory, model, optimizer)
        total_reward += reward
        state = next_state

        if done or 0 < timeout < t:
            break

    controller.stop(save=False)
    return max(epsilon * 0.99, 0.05)



def run(rom_path, device, SCALE_FACTOR, USE_GRAYSCALE,  timeout, num_episodes=100, batch_size=128, epsilon=1.0 ):

    controller = Controller(rom_path)
    screen_size = controller.screen_size()
    controller.stop()
    model = DQN(int(screen_size[0] * SCALE_FACTOR), int(screen_size[1] * SCALE_FACTOR), len(controller.movements), USE_GRAYSCALE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    memory = ReplayMemory(1000)

    for i_episode in tqdm(
        range(num_episodes)
    ):
       epsilon = run_episode( rom_path, model, memory, optimizer, epsilon, device, SCALE_FACTOR, USE_GRAYSCALE, timeout, batch_size)

    torch.save(model.state_dict(), "./checkpoints/pokemon_rl_model_final.pth")
