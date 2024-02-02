# -*- coding: utf-8 -*-
import torch
from itertools import count
from torch.distributions import Categorical
from controls import Controller  # Ensure this is your actual controller
from utils import (
    image_to_tensor,
    calc_rewards,
)  # Assuming these are your utility functions
import multiprocessing
from functools import partial
from PPO import PPOBuffer, ppo_update, PolicyNetwork, ValueNetwork


def run_episode_ppo(
    episode_num,
    rom_path,
    state_path,
    policy_net,
    value_net,
    device,
    ppo_buffer,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    timeout,
    gamma=0.99,
    tau=0.95,
    document_mode=False,
):
    controller = Controller(rom_path, state_path)
    controller.handleMovement("A")  # Start the game
    state = image_to_tensor(
        controller.screen_image(), device, SCALE_FACTOR, USE_GRAYSCALE
    )

    for t in count():
        # Select action based on the current policy
        state_tensor = state.unsqueeze(0)  # Add batch dimension
        action_probs = policy_net(state_tensor)
        value = value_net(state_tensor)
        m = Categorical(action_probs)
        action = m.sample()

        # Perform action in env
        controller.handleMovement(controller.movements[action.item()])
        reward = calc_rewards(
            controller
        )  # Ensure this function is implemented correctly
        next_state = image_to_tensor(
            controller.screen_image(), device, SCALE_FACTOR, USE_GRAYSCALE
        )
        is_terminal = float(
            t + 1 == timeout
        )  # Adjust based on your termination conditions

        # Store experiences in PPOBuffer
        ppo_buffer.add(state, action, m.log_prob(action), reward, value, is_terminal)

        state = next_state
        if is_terminal:
            break

    controller.stop(save=False)


def collect_experiences(
    rom_path,
    state_path,
    policy_net,
    value_net,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    timeout,
    num_episodes,
    cpus,
):
    ppo_buffer = PPOBuffer()

    # Define a partial function for ease of multiprocessing
    run_episode_partial = partial(
        run_episode_ppo,
        rom_path=rom_path,
        state_path=state_path,
        policy_net=policy_net,
        value_net=value_net,
        device=device,
        ppo_buffer=ppo_buffer,
        SCALE_FACTOR=SCALE_FACTOR,
        USE_GRAYSCALE=USE_GRAYSCALE,
        timeout=timeout,
    )

    with multiprocessing.Pool(processes=cpus) as pool:
        pool.map(run_episode_partial, range(num_episodes))

    # Now ppo_buffer contains experiences from all episodes
    return ppo_buffer


def ppo_train(
    rom_path,
    state_paths,
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    num_episodes,
    timeout,
    batch_size,
    cpus,
):
    for state_path in state_paths:
        # Collect experiences
        ppo_buffer = collect_experiences(
            rom_path,
            state_path,
            policy_net,
            value_net,
            device,
            SCALE_FACTOR,
            USE_GRAYSCALE,
            timeout,
            num_episodes,
            cpus,
        )

        # Prepare data for PPO update
        (
            states,
            actions,
            log_probs,
            rewards,
            values,
            is_terminals,
        ) = ppo_buffer.get_batch()

        # Perform PPO update
        ppo_update(
            policy_net,
            value_net,
            optimizer_policy,
            optimizer_value,
            states,
            actions,
            log_probs,
            rewards,
            values,
            is_terminals,
            device,
        )

        # Save models and log results as needed


def run_training(
    rom_path,
    state_paths,
    input_size,
    hidden_size,
    action_size,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    num_episodes,
    episodes_per_batch,
    timeout,
    cpus,
    device,
):
    # Initialize the device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Initialize models
    policy_net = PolicyNetwork(input_size, hidden_size, action_size).to(device)
    value_net = ValueNetwork(input_size, hidden_size).to(device)

    # Initialize optimizers
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=1e-3)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)

    # Run PPO Training
    ppo_train(
        rom_path=rom_path,
        state_paths=state_paths,
        policy_net=policy_net,
        value_net=value_net,
        optimizer_policy=optimizer_policy,
        optimizer_value=optimizer_value,
        device=device,
        SCALE_FACTOR=SCALE_FACTOR,
        USE_GRAYSCALE=USE_GRAYSCALE,
        num_episodes=num_episodes,
        timeout=timeout,
        batch_size=episodes_per_batch,  # This parameter might need to be adjusted based on your ppo_update function
        cpus=cpus,
    )

    # After training, save your models
    torch.save(policy_net.state_dict(), "policy_net.pth")
    torch.save(value_net.state_dict(), "value_net.pth")


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
    cpus=cpus,
):
    screen_size = Controller(rom_path).screen_size()

    input_size = 1024  # Adjust based on your input dimension
    hidden_size = 512  # Adjust based on your preference
    action_size = 4
    SCALE_FACTOR = 0.5
    USE_GRAYSCALE = True
    num_episodes = 1000
    episodes_per_batch = 20
    timeout = 1000
    cpus = 4
    device = "mps"

    run_training(
        rom_path,
        state_paths,
        input_size,
        hidden_size,
        action_size,
        SCALE_FACTOR,
        USE_GRAYSCALE,
        num_episodes,
        episodes_per_batch,
        timeout,
        cpus,
        device,
    )
