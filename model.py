# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from itertools import count
import os
from PIL import Image
from controls import Controller
import multiprocessing
from multiprocessing import Pool
from itertools import count
from collections import namedtuple
from tqdm import tqdm
from TorchModel import DQN, ReplayMemory
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))



def run_model(
    rom_path,
    locations,
    location_address,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    goal_locs,
    goal_targets,
    timeout,
    num_episodes=15,
    report_interval=10,
    num_workers=None,
):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    controller = Controller(rom_path)
    movements = controller.movements
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
    screen_size = controller.screen_size()
    scaled_size = (
        int(screen_size[0] * SCALE_FACTOR),
        int(screen_size[1] * SCALE_FACTOR),
    )
    shared_model = DQN(
        scaled_size[0], scaled_size[1], len(movements), USE_GRAYSCALE
    ).to(device)
    shared_model.share_memory()  # Prepare model for sharing across processes
    shared_memory = ReplayMemory(1000)
    shared_optimizer = optim.Adam(shared_model.parameters(), lr=0.005)
    default_reward = 0.1
    phase = 0
    # Load checkpoint if available
    checkpoint_path = "./checkpoints/pokemon_rl_checkpoint.pth"
    epsilon_initial = 0.9
    epsilon_decay = 0.99
    epsilon_min = 0.1
    start_episode, epsilon_initial = load_checkpoint(
        checkpoint_path, shared_model, shared_optimizer, epsilon_initial
    )
    gamma = 0.5
    batch_size = 128
    epsilon = epsilon_initial
    min_update_steps = 5
    results = []
    for start in tqdm(range(0, num_episodes, report_interval)):
        end = min(start + report_interval, num_episodes)
        episodes_this_round = end - start
        episodes_per_worker = episodes_this_round // num_workers

        args = [
            (
                worker_id,
                episodes_per_worker,
                rom_path,
                locations,
                location_address,
                device,
                SCALE_FACTOR,
                USE_GRAYSCALE,
                goal_locs,
                timeout,
                shared_model.state_dict(),
                epsilon,
                negative_keywords,
                positive_keywords,
                default_reward,
                phase,
                movements,
            )
            for worker_id in range(num_workers)
        ]

        with Pool(num_workers) as pool:
            partial_results = pool.starmap(run_episode, args)
            # extract run times from partial results
            run_times = [result[1] for result in partial_results]
            # extract experiences from partial results
            partial_results = [result[0] for result in partial_results]
            new_experiences = sum(
                len(worker_result) for worker_result in partial_results
            )
        results.extend(partial_results)
        update_steps = max(min_update_steps, new_experiences // batch_size)

        # Aggregate and report results after every 'report_interval' episodes
        aggregate_results(
            results,
            shared_memory,
            shared_optimizer,
            shared_model,
            device,
            gamma,
            batch_size,
            update_steps,
        )
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # save_checkpoint(
        #     {
        #         "epoch": start + report_interval,
        #         "state_dict": shared_model.state_dict(),
        #         "optimizer": shared_optimizer.state_dict(),
        #         "epsilon": epsilon,
        #     },
        #     filename=f"./checkpoints/pokemon_rl_checkpoint_{start + report_interval + start_episode}.pth",
        # )
        # Print the average run time per worker
        print(f"Average run time per worker: {np.mean(run_times)}")

    return results


def image_to_tensor(image, SCALE_FACTOR, USE_GRAYSCALE, device):
    # If image is already a tensor, just ensure it's on the correct device and return
    if torch.is_tensor(image):
        return image.to(device)

    # If image is a NumPy array, convert it to a tensor
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    if SCALE_FACTOR != 1:
        image = image.resize([int(s * SCALE_FACTOR) for s in image.size])

    if USE_GRAYSCALE:
        image = image.convert("L")

    image = np.array(image)

    if USE_GRAYSCALE:
        image = np.expand_dims(image, axis=2)

    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32) / 255
    return image.to(device)



def select_action(state, epsilon, movements, model, device):
    if random.random() > epsilon:
        with torch.no_grad():
            return (
                model(state).max(1)[1].view(1, 1).to(device)
            )  # Sending tensor to the specified device
    else:
        return torch.tensor(
            [[random.randrange(len(movements))]],
            dtype=torch.long,
            device=device,
        )  # Sending tensor to the specified device


def optimize_model(batch, model, optimizer, device, gamma):
    criterion = nn.SmoothL1Loss()
    optimizer.zero_grad()

    total_loss = 0.0
    print("Batch length:", len(batch))
    for sub_batch in batch:
        if not sub_batch:
            continue

        state_batch, action_batch, reward_batch, next_state_batch, non_final_mask = _transition_to_batch(sub_batch, device)

        if not state_batch:
            continue

        if state_batch:
           #Concatenate all tensors in the list along the batch dimension
            state_batch = torch.cat(state_batch, dim=0)

        # Compute Q-values for the current states
        # check shape of state_batch
        print("State batch shape: ")


        print("Finished printing state batch shape")
        if state_batch.nelement() == 0:
            continue

        q_values = model(state_batch)

        # # Compute Q-values for the actions taken
        # print("q_values shape:", q_values.shape)
        # print("action_batch shape:", action_batch.shape)


        action_batch = action_batch.unsqueeze(-1)

# Perform gather operation
        state_action_values = q_values.gather(1, action_batch)

        # Compute V-values for the next states
        if next_state_batch:
            next_state_batch = torch.cat(next_state_batch, dim=0)
            next_state_values = torch.zeros(len(state_batch), device=device)
            next_state_values[non_final_mask] = model(next_state_batch).max(1)[0].detach()
        else:
            next_state_values = torch.zeros(len(state_batch), device=device)

        # Compute the expected Q-values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute loss
        expected_state_action_values = expected_state_action_values.unsqueeze(1)
        loss = criterion(state_action_values, expected_state_action_values)

        # Accumulate the loss
        total_loss += loss.item()
        loss.backward()

    # Update the model parameters
    optimizer.step()

    # Return the average loss for monitoring (optional)
    return total_loss / len(batch) if len(batch) > 0 else 0



def rewards(
    action,
    loc,
    visited_locations,
    controller,
    positive_keywords,
    negative_keywords,
    default_reward,
):
    # store if has been rewarded recently
    # if has been rewarded recently, then don't reward again

    total_reward = 0
    text = controller.get_text_on_screen()
    # check if any of the positive keywords are in the text
    for keyword in positive_keywords:
        if keyword in text.lower() and not positive_keywords[keyword]:
            positive_keywords[keyword] = True
            total_reward += default_reward


        else:
            positive_keywords[keyword] = False
    # check if any of the negative keywords are in the text
    for keyword in negative_keywords:
        if keyword in text.lower() and not negative_keywords[keyword]:
            negative_keywords[keyword] = True
            total_reward -= default_reward

        else:
            negative_keywords[keyword] = False

    # We should discourage start and select
    if action == "START" or action == "SELECT":
        total_reward -= default_reward * 2

    # Encourage exploration
    # if loc isn't in set then reward
    if loc not in visited_locations:
        total_reward += default_reward
        # add loc to visited locations
        visited_locations.add(loc)

    else:
        # Discourage  revisiting locations too much
        total_reward -= default_reward


    return total_reward


def aggregate_results(
    results, memory, optimizer, model, device, gamma, batch_size, update_steps
):
    # Unpack results and update shared replay memory
    for worker_experiences in results:
        for experience in worker_experiences:
            memory.push(*experience)

    # Update the model based on the aggregated experiences
    if len(memory) > batch_size:
        for _ in range(update_steps):
            variable_batches = memory.sample_variable(batch_size)
            optimize_model(variable_batches, model, optimizer, device, gamma)



def _transition_to_batch(transitions, device):
    batch = Transition(*zip(*transitions))
    state_batch = [s.to(device) for s in batch.state if s is not None]
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=device)
    reward_batch = torch.tensor(batch.reward, device=device)
    next_state_batch = [s.to(device) for s in batch.next_state if s is not None]
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=device)

    return state_batch, action_batch, reward_batch, next_state_batch, non_final_mask

def run_episode(
    worker_id,
    num_episodes,
    rom_path,
    locations,
    location_address,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    goal_locs,
    timeout,
    shared_model_state,
    epsilon,
    negative_keywords,
    positive_keywords,
    default_reward,
    phase,
    movements,
):
    controller = Controller(rom_path)
    screen_size = controller.screen_size()
    scaled_size = (
        int(screen_size[0] * SCALE_FACTOR),
        int(screen_size[1] * SCALE_FACTOR),
    )
    model = DQN(
        scaled_size[0], scaled_size[1], len(controller.movements), USE_GRAYSCALE
    ).to(device)
    model.load_state_dict(shared_model_state)  # Load shared model state
    # add a random number to the timeout  
    time_per_episode = []
    experiences = []

    for i_episode in range(num_episodes):
        controller.stop()
        controller = Controller(rom_path)
        state = image_to_tensor(
            controller.screen_image(), SCALE_FACTOR, USE_GRAYSCALE, device
        )
        visited_locations = set()
        total_reward = 0

        for t in count():
            action = select_action(state, epsilon, movements, model, device)
            controller.handleMovement(controller.movements[action.item()])
            reward = torch.tensor([-0.01], dtype=torch.float32, device=device)
            img = controller.screen_image()
            loc = controller.get_memory_value(location_address)
            done = False

            reward_value = rewards(
                action,
                loc,
                visited_locations,
                controller,
                positive_keywords,
                negative_keywords,
                default_reward,
            )
            reward += reward_value
            if loc in locations and locations[loc] == goal_locs[phase]:
                reward += 1
                done = True

            next_state = (
                image_to_tensor(img, SCALE_FACTOR, USE_GRAYSCALE, device)
                if not done
                else None
            )
            experiences.append(Transition(state, action, reward, next_state))

            # for testing randomly add another experience
            if random.random() > 0.5:
                experiences.append(Transition(state, action, reward, next_state))

            total_reward += reward.item()
            state = next_state

            if done or (timeout > 0 and t > timeout):
                controller.stop()  # free up some memory
                break

        time_per_episode.append(t)

    return (experiences, time_per_episode)


def save_checkpoint(state, filename="pokemon_rl_checkpoint.pth"):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer, epsilon):
    start_episode = 0
    # Check for latest checkpoint in checkpoints folder
    tmp_path = checkpoint_path
    if os.path.isdir("./checkpoints"):
        checkpoints = [f for f in os.listdir("./checkpoints") if f.endswith(".pth")]
        if len(checkpoints) > 0:
            # sort checkpoints by last modified date
            checkpoints.sort(key=lambda x: os.path.getmtime("./checkpoints/" + x))
            checkpoint_path = "./checkpoints/" + checkpoints[-1]
            # Set the start episode to the number of the checkpoint
            start_episode = int(checkpoint_path.split("_")[-1][:-4])
    else:
        os.mkdir("./checkpoints")
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_episode = checkpoint["epoch"]
            epsilon = checkpoint["epsilon"]
        except:
            print("Failed to load checkpoint")
        checkpoint_path = tmp_path
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")

    return start_episode, epsilon