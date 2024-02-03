# -*- coding: utf-8 -*-
import torch
from itertools import count
from torch.distributions import Categorical
from controls import Controller  # Ensure this is your actual controller
from utils import image_to_tensor, document
import multiprocessing
from functools import partial
from PPO import PPOBuffer, ppo_update, PolicyNetwork, ValueNetwork
from rewards import calc_rewards
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='training_log.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def run_episode_ppo(
    episode_num,
    rom_path,
    state_path,
    policy_net,
    value_net,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    timeout,
    gamma=0.99,
    tau=0.95,
    document_mode=False,
):
    # Initialize controller and a local buffer for this episode
    controller = Controller(rom_path, state_path)
    controller.handleMovement("A")  # Start the game
    local_buffer = PPOBuffer()  # Create a local buffer
    
    state = image_to_tensor(controller.screen_image(), device, SCALE_FACTOR, USE_GRAYSCALE)
    locs = set()
    xy = set()
    imgs = []
    max_total_level = [0]
    max_total_exp = [0]

    for t in count():
        # Select action based on the current policy
        action_probs = policy_net(state)
        value = value_net(state)
        m = Categorical(action_probs)
        action = m.sample()

        # Perform action in env
        controller.handleMovement(controller.movements[action.item()])
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
        next_state = image_to_tensor(img, device, SCALE_FACTOR, USE_GRAYSCALE)
        is_terminal = float(t + 1 == timeout)

        # Store experiences in the local PPOBuffer
        local_buffer.add(state, action, m.log_prob(action), reward, value, is_terminal)

        state = next_state
        if is_terminal:
            break
        
        if document_mode:
            # Assuming document is a function that logs or saves episode data for review
            document(episode_num, t, img, action, reward, timeout, 1, "ppo")

    controller.stop(save=False)  # Stop the controller, without saving the game state

    return local_buffer

def iterate_batches(data, batch_size):
    """Yield successive n-sized chunks from data."""
    for i in range(0, len(data[0]), batch_size):
        yield (d[i:i + batch_size] for d in data)

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
    gamma=0.99,
    tau=0.95,
):
    training_info = {'episode_rewards': [], 'losses': []}  # For storing training info
    main_buffer = PPOBuffer()  # Main buffer to hold merged data from all episodes

    for state_path in state_paths:
        # Collect experiences from multiple episodes in parallel
        run_episode_partial = partial(
            run_episode_ppo,
            rom_path=rom_path,
            state_path=state_path,
            policy_net=policy_net,
            value_net=value_net,
            device=device,
            SCALE_FACTOR=SCALE_FACTOR,
            USE_GRAYSCALE=USE_GRAYSCALE,
            timeout=timeout,
            gamma=gamma,
            tau=tau,
            document_mode=False,  # Set according to your needs
        )

        with multiprocessing.Pool(processes=cpus) as pool:
            episode_buffers = pool.imap(run_episode_partial, range(num_episodes))
            for episode_buffer in tqdm(episode_buffers, total=num_episodes, desc="Collecting Experiences"):
                main_buffer.merge(episode_buffer)

    # After collecting and merging data, compute GAE and returns for the main buffer
    last_value = 0  # Assuming last_value is obtained here
    main_buffer.compute_gae_and_returns(last_value, gamma=gamma, tau=tau)

    # Manual batching and training
    states, actions, log_probs_old, returns, advantages, is_terminal = main_buffer.get_batch()
    policy_losses, value_losses = [], []

    # Manually iterate over batches
    total_size = states.size(0)
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        states_batch = states[start:end]
        actions_batch = actions[start:end]
        log_probs_old_batch = log_probs_old[start:end]
        returns_batch = returns[start:end]
        advantages_batch = advantages[start:end]

        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        # Update policy and value networks
        loss_info = ppo_update(
            policy_net,
            value_net,
            optimizer_policy,
            optimizer_value,
            states_batch,
            actions_batch,
            log_probs_old_batch,
            returns_batch,
            advantages_batch,
            device=device,
        )
        policy_losses.append(loss_info['policy_loss'])
        value_losses.append(loss_info['value_loss'])

    # Logging
    avg_policy_loss = np.mean(policy_losses)
    avg_value_loss = np.mean(value_losses)
    logging.info(f'Average Policy Loss: {avg_policy_loss}, Average Value Loss: {avg_value_loss}')


def run_training(
    rom_path,
    state_paths,  # This should be a list
    SCALE_FACTOR,
    USE_GRAYSCALE,
    num_episodes,
    episodes_per_batch,
    timeout,
    cpus,
    device,
    gamma=0.99,
    tau=0.95,
):

    controller = Controller(rom_path)
    screen_height, screen_width = controller.screen_size()
    screen_height, screen_width = int(screen_height * SCALE_FACTOR), int(screen_width * SCALE_FACTOR)
    action_size = len(controller.movements)
    policy_net = PolicyNetwork(screen_height, screen_width, action_size, USE_GRAYSCALE).to(device)
    value_net = ValueNetwork(screen_height, screen_width, USE_GRAYSCALE).to(device)

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=1e-3)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)
    print("Starting training...")
    # Proceed with the training
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
        batch_size=episodes_per_batch,  # Adjust based on your setup
        cpus=cpus,
        gamma=gamma,
        tau=tau,
    )


    torch.save(policy_net.state_dict(), "policy_net.pth")
    torch.save(value_net.state_dict(), "value_net.pth")

def run(
    rom_path="Pokemon - Crystal Version.gbc",
    device="cpu",
    SCALE_FACTOR=1,
    USE_GRAYSCALE=False,
    state_paths=["./states/start.state"],  # This should be a list to accommodate multiple paths
    num_episodes=10,
    episodes_per_batch=1,
    timeout=10,
    cpus=4
):

    # Run training with adjusted parameters
    run_training(
        rom_path=rom_path,
        state_paths=state_paths,  # Ensure this is passed as a list
        SCALE_FACTOR=SCALE_FACTOR,
        USE_GRAYSCALE=USE_GRAYSCALE,
        num_episodes=num_episodes,
        episodes_per_batch=episodes_per_batch,
        timeout=timeout,
        cpus=cpus,
        device=device,
    )

if __name__ == "__main__":
    run()

