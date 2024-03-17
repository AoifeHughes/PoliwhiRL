# -*- coding: utf-8 -*-
import os
import torch.optim as optim
import time
import json
from PoliwhiRL.environment.controller import Controller
from .rainbowDQN import RainbowDQN
from .replaybuffer import PrioritizedReplayBuffer
from .utils import save_checkpoint, load_checkpoint
from .singlerainbow import run as run_single
from .doublerainbow import run as run_rainbow_parallel
from PoliwhiRL.utils import plot_best_attempts


def setup_environment(config):
    """
    Sets up the environment based on the provided configuration.
    """
    if config['reward_image_folder'] is not None:
        fldr = config['reward_image_folder']
        image_files = [f'{fldr}/{f}' for f in os.listdir(config['reward_image_folder'])]
    else:
        image_files = []
    return Controller(
        config["rom_path"],
        config["state_path"],
        timeout=config["episode_length"],
        log_path="./logs/rainbow_env.json",
        use_sight=config["sight"],
        extra_files=config["extra_files"],
        reward_locations_xy=config["reward_locations_xy"],
        scaling_factor=config["scaling_factor"],
        use_grayscale=config["use_grayscale"],
        reward_images=image_files,
    )


def initialize_training(config, env):
    """
    Initializes training components such as the policy and target networks, optimizer, and replay buffer.
    """
    input_shape = (1 if config["use_grayscale"] else 3, *env.screen_size())
    policy_net = RainbowDQN(input_shape, len(env.action_space), config["device"]).to(
        config["device"]
    )
    target_net = RainbowDQN(input_shape, len(env.action_space), config["device"]).to(
        config["device"]
    )
    optimizer = optim.Adam(policy_net.parameters(), lr=config["learning_rate"])
    replay_buffer = PrioritizedReplayBuffer(config["capacity"], config["alpha"])

    # Load checkpoint if available and update training components accordingly
    checkpoint = load_checkpoint(config)
    if checkpoint:
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        target_net.load_state_dict(checkpoint["target_net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        replay_buffer.load_state_dict(checkpoint["replay_buffer_state_dict"])
        # Ensure the target_net is in eval mode
        target_net.eval()

    return policy_net, target_net, optimizer, replay_buffer


def run_training(config, env, policy_net, target_net, optimizer, replay_buffer):
    """
    Runs the training loop, either in single or parallel mode based on configuration.
    """
    if not config["run_parallel"]:
        return run_single(config, env, policy_net, target_net, optimizer, replay_buffer)
    else:
        return run_rainbow_parallel(
            config, policy_net, target_net, optimizer, replay_buffer
        )


def finalize_training(
    config,
    start_time,
    rewards,
    total_losses,
    total_td_errors,
    total_beta_values,
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
):
    """
    Finalizes the training process by logging data, plotting results, and saving a final checkpoint.
    """
    total_time = time.time() - start_time
    log_data = {
        "total_time": total_time,
        "average_reward": sum(rewards) / len(rewards) if rewards else 0,
        "losses": total_losses,
        "beta_values": total_beta_values,
        "td_errors": total_td_errors,
    }
    # Ensure logs directory exists
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/training_log.json", "w") as outfile:
        json.dump(log_data, outfile, indent=4)
    print("Training log saved to ./logs/training_log.json")
    # Plot results
    plot_best_attempts(
        "./results/",
        config["num_episodes"],
        f"RainbowDQN_{config['run_parallel']}_final",
        rewards,
    )
    # Save checkpoint
    save_checkpoint(
        config,
        policy_net,
        target_net,
        optimizer,
        replay_buffer,
        rewards,
    )


def run(**config):
    """
    Main function to run the training process.
    """
    start_time = time.time()
    env = setup_environment(config)
    policy_net, target_net, optimizer, replay_buffer = initialize_training(config, env)
    # Start training
    (total_losses, total_beta_values, total_td_errors, rewards) = run_training(
        config, env, policy_net, target_net, optimizer, replay_buffer
    )
    # Finalize training
    finalize_training(
        config,
        start_time,
        rewards,
        total_losses,
        total_td_errors,
        total_beta_values,
        policy_net,
        target_net,
        optimizer,
        replay_buffer,
    )
