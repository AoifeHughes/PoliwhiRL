# -*- coding: utf-8 -*-
import torch
from itertools import count
from torch.distributions import Categorical
from PoliwhiRL.env.controls import Controller  # Ensure this is your actual controller
from PoliwhiRL.utils.utils import image_to_tensor
import multiprocessing
from functools import partial
from PPO import PPOBuffer, ppo_update, PolicyNetwork, ValueNetwork
from PoliwhiRL.env.rewards import calc_rewards
import torch.optim as optim
from tqdm import tqdm

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
        next_state = image_to_tensor(
            img, device, SCALE_FACTOR, USE_GRAYSCALE
        )
        is_terminal = float(
            t + 1 == timeout
        )  

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
        list(tqdm(pool.imap(run_episode_partial, range(num_episodes)), total=num_episodes))


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
    gamma=0.99,
    tau=0.95,
):
    for state_path in state_paths:
        # Collect experiences
        ppo_buffer = PPOBuffer()
        
        # Define a partial function for multiprocessing
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
            gamma=gamma,  # Note: These are not used inside run_episode_ppo directly but could be if adjusting the function
            tau=tau,
        )
        
        with multiprocessing.Pool(processes=cpus) as pool:
            pool.map(run_episode_partial, range(num_episodes))

        # After collecting experiences, compute GAE and returns before updating
        # Assuming there's a method to get the last state's value or setting it to 0 if terminal
        # This part might need adjustment based on how you handle episode ends and next state value
        last_value = 0  # This should be replaced with an actual value computation if necessary
        
        ppo_buffer.compute_gae_and_returns(last_value, gamma=gamma, tau=tau)

        # Get batches for training
        data_loader = torch.utils.data.DataLoader(
            ppo_buffer.get_batch(),
            batch_size=batch_size,
            shuffle=True,
        )

        for states, actions, log_probs_old, returns, advantages in data_loader:
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ppo_update(
                policy_net,
                value_net,
                optimizer_policy,
                optimizer_value,
                states,
                actions,
                log_probs_old,
                returns,
                advantages,
                device=device,
            )

        # Optional: Log training progress, save models, etc.

def run_training(
    rom_path,
    state_paths,  # This should be a list
    action_size,
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

    policy_net = PolicyNetwork(screen_height, screen_width, action_size, USE_GRAYSCALE).to(device)
    value_net = ValueNetwork(screen_height, screen_width, USE_GRAYSCALE).to(device)

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=1e-3)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)

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
    num_episodes=100,
    episodes_per_batch=20,
    timeout=1000,
    cpus=4
):

    # Run training with adjusted parameters
    run_training(
        rom_path=rom_path,
        state_paths=state_paths,  # Ensure this is passed as a list
        action_size=4,  # Adjust based on the game's action space
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

