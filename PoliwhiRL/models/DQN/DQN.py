# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PoliwhiRL.environment.controls import Controller
from PoliwhiRL.utils.utils import (
    save_results,
    plot_best_attempts,
    load_checkpoint,
    save_checkpoint,
)
import torch.optim as optim
import multiprocessing
from PoliwhiRL.utils.utils import image_to_tensor, select_action
from itertools import count
from PoliwhiRL.utils.utils import document
from PoliwhiRL.environment.rewards import calc_rewards
from PoliwhiRL.models.DQN.ReplayMemory import ReplayMemory
import random


class DQN(nn.Module):
    def __init__(self, h, w, outputs, USE_GRAYSCALE):
        super(DQN, self).__init__()
        self.USE_GRAYSCALE = USE_GRAYSCALE
        # Convolutional layers
        self.conv1 = nn.Conv2d(1 if USE_GRAYSCALE else 3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2
        )  # Additional convolutional layer
        self.bn4 = nn.BatchNorm2d(64)

        self._to_linear = None
        self._compute_conv_output_size(h, w)
        self.fc1 = nn.Linear(self._to_linear, 512)  # Larger fully connected layer
        self.fc2 = nn.Linear(512, outputs)  # Additional fully connected layer

    def _compute_conv_output_size(self, h, w):
        x = torch.rand(1, 1 if self.USE_GRAYSCALE else 3, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def optimize_model(
    batch_size,
    device,
    memory,
    primary_model,
    target_model,
    optimizer,
    GAMMA=0.9,
    n_steps=5,
):
    # Sample a batch of n-step sequences
    sequences = memory.sample(batch_size)

    # Initialize lists for states, actions, rewards, and next states
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    non_final_mask = []

    for sequence in sequences:
        s = sequence[0]
        # Use torch operations for clipping and calculating cumulative reward to keep computation on the device
        cumulative_reward = sum(
            (GAMMA**i) * torch.clamp(s[i][2], -1, 1) for i in range(len(s))
        )
        reward_batch.append(cumulative_reward)
        state_batch.append(s[0][0])
        action_batch.append(s[0][1])
        next_state = s[-1][3] if s[-1][3] is not None else None
        next_state_batch.append(next_state)
        non_final_mask.append(next_state is not None)

    # Convert lists to tensors
    state_batch = torch.cat(state_batch).to(device)
    action_batch = torch.cat(action_batch).to(device)
    reward_batch = torch.tensor(reward_batch, device=device)
    non_final_next_states = torch.cat(
        [s for s in next_state_batch if s is not None]
    ).to(device)
    non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.bool)

    # Compute Q(s_t, a) using the primary_model
    state_action_values = primary_model(state_batch).gather(1, action_batch)

    # Initialize next state values to zero
    next_state_values = torch.zeros(batch_size, device=device)
    # Compute V(s_{t+n}) for all next states using the target_model
    if non_final_mask.sum() > 0:
        next_state_values[non_final_mask] = (
            target_model(non_final_next_states).max(1)[0].detach()
        )

    # Compute the expected Q values
    expected_state_action_values = (
        next_state_values * (GAMMA**n_steps)
    ) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in primary_model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def run(
    rom_path,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    episode_length,
    state_paths,
    num_episodes,
    episodes_per_batch,
    batch_size,
    nsteps,
    cpus=8,
    explore_mode=False,
):
    if explore_mode:
        explore_episode(rom_path, episode_length, nsteps)
        return
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
        100000, n_steps=nsteps, multiCPU=device == torch.device("cpu")
    )
    start_episode, init_epsilon = load_checkpoint(
        "./checkpoints/", primary_model, optimizer, 0, 1.0
    )

    results = run_phase(
        init_epsilon,
        1,
        0.1,
        num_episodes,
        episodes_per_batch,
        batch_size,
        episode_length,
        rom_path,
        state_paths,
        primary_model,
        target_model,  # Pass target model to run_phase
        memory,
        optimizer,
        device,
        SCALE_FACTOR,
        USE_GRAYSCALE,
        nsteps,
        checkpoint=True,
        phase=f"{episode_length}",
        cpus=cpus,
        start_episode=start_episode,
    )

    save_checkpoint(
        "./checkpoints/", primary_model, optimizer, num_episodes, 0.1, episode_length
    )
    start_episode += num_episodes
    save_results("./results/", start_episode, results)


def run_phase(
    init_epsilon,
    epsilon_max,
    epsilon_min,
    num_episodes,
    episodes_per_batch,
    batch_size,
    timeout,
    rom_path,
    state_paths,
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
                state_paths,
                primary_model,
                target_model,
                epsilons_batch[i - batch_start],
                device,
                memory,
                optimizer,
                SCALE_FACTOR,
                USE_GRAYSCALE,
                timeout,
                batch_size,
                n_steps,
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


def explore_episode(rom_path, timeout, nsteps, state_paths=None):
    if isinstance(state_paths, list):
        state_path = np.random.choice(state_paths)
    else:
        state_path = None
    controller = Controller(rom_path, state_path)
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
                timeout,
                1,
                "explore",
            )
            stored_states += 1
        if len(states) > nsteps:
            states.pop(0)


def soft_update(target_model, primary_model, tau=0.001):
    for target_param, primary_param in zip(
        target_model.parameters(), primary_model.parameters()
    ):
        target_param.data.copy_(
            tau * primary_param.data + (1.0 - tau) * target_param.data
        )


def run_episode(
    i,
    rom_path,
    state_paths,
    primary_model,
    target_model,
    epsilon,
    device,
    memory,
    optimizer,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    timeout,
    batch_size,
    n_steps,
    phase,
    document_mode=False,
):
    if isinstance(state_paths, list):
        state_path = np.random.choice(state_paths)
    else:
        state_path = None
    controller = Controller(rom_path, state_path)
    movements = controller.movements
    initial_img = controller.screen_image()
    state = image_to_tensor(initial_img, device, SCALE_FACTOR, USE_GRAYSCALE)
    n_step_buffer = []
    total_reward = 0
    max_total_level = [0]
    max_total_exp = [0]
    locs = set()
    xy = set()
    imgs = []
    controller.handleMovement("A")  # Start the game

    for t in count():
        action = select_action(state, epsilon, device, movements, primary_model)
        controller.handleMovement(movements[action.item()])
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
        action_tensor = torch.tensor([[action]], device=device)
        reward_tensor = torch.tensor([reward], device=device)

        next_state = image_to_tensor(img, device, SCALE_FACTOR, USE_GRAYSCALE)

        n_step_buffer.append((state, action_tensor, reward_tensor, next_state))

        if len(n_step_buffer) == n_steps:
            # Add the n-step buffer to memory
            memory.push(n_step_buffer.copy())
            n_step_buffer.clear()
            # Optionally optimize the model here or after collecting more experience
            if len(memory) >= batch_size:
                optimize_model(
                    batch_size,
                    device,
                    memory,
                    primary_model,
                    target_model,
                    optimizer,
                    GAMMA=0.9,
                    n_steps=n_steps,
                )
                if i % 10 == 0:
                    soft_update(target_model, primary_model, tau=0.001)

        state = next_state
        total_reward += reward
        if timeout and t >= timeout:
            break
        if document_mode:
            document(
                i,
                t,
                img,
                movements[action.item()],
                reward,
                timeout,
                epsilon,
                phase,
            )

    controller.stop(save=False)
    print(f"Episode {i} finished after {t} timesteps with reward {total_reward}")
    return total_reward
