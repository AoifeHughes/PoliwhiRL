# -*- coding: utf-8 -*-
import os
import torch.optim as optim
import time
import json
from PoliwhiRL.environment.controller import Controller
from .rainbowDQN import RainbowDQN
from .replaybuffer import PrioritizedReplayBuffer
from .utils import save_checkpoint, load_checkpoint, epsilon_by_frame
from .singlerainbow import run as run_single
from .doublerainbow import run as run_rainbow_parallel
from PoliwhiRL.utils import plot_best_attempts


def run(
    rom_path,
    state_path,
    episode_length,
    device,
    num_episodes,
    batch_size,
    checkpoint_path="rainbow_checkpoint.pth",
    run_parallel=False,
    sight=False,
    runs_per_worker=100,
    num_workers=8,
    memories=0,
    checkpoint_interval=100,
    epsilon_by_location=False,
    extra_files=[],
    reward_locations_xy={},
):
    start_time = time.time()  # For computational efficiency tracking
    env = Controller(
        rom_path,
        state_path,
        timeout=episode_length,
        log_path="./logs/rainbow_env.json",
        use_sight=sight,
        extra_files=extra_files,
        reward_locations_xy=reward_locations_xy,
    )
    gamma = 0.99
    alpha = 0.6
    beta_start = 0.4
    beta_frames = 1000
    frame_idx = 0
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 50000
    learning_rate = 1e-4
    capacity = 50000
    update_target_every = 1000
    reward_threshold = 0.3
    reward_sensitivity = 0.15
    reward_window_size = 10
    losses = []
    epsilon_values = []  # Tracking epsilon values for exploration metrics
    beta_values = []  # For priority buffer metrics
    td_errors = []  # For DQN metrics
    rewards = []
    screen_size = env.screen_size()
    input_shape = (3, int(screen_size[0]), int(screen_size[1]))
    policy_net = RainbowDQN(input_shape, len(env.action_space), device).to(device)
    target_net = RainbowDQN(input_shape, len(env.action_space), device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = PrioritizedReplayBuffer(capacity, alpha)
    frames_in_loc = {i: 0 for i in range(256)}
    epsilons_by_location = {i: 1.0 for i in range(256)}

    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint is not None:
        start_episode = checkpoint["episode"] + 1
        frame_idx = checkpoint["frame_idx"]
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        target_net.load_state_dict(checkpoint["target_net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        frames_in_loc = checkpoint["frames_in_loc"]
        rewards = checkpoint["rewards"]
        epsilons_by_location = checkpoint["epsilon_by_location"]
    else:
        start_episode = 0

    if not run_parallel:
        losses, rewards_n, memories = run_single(
            start_episode,
            num_episodes,
            env,
            device,
            policy_net,
            target_net,
            optimizer,
            replay_buffer,
            checkpoint_path,
            frame_idx,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            beta_start,
            beta_frames,
            batch_size,
            gamma,
            update_target_every,
            losses,
            epsilon_values,
            beta_values,
            td_errors,
            rewards,
            checkpoint_interval,
            epsilon_by_location,
            frames_in_loc,
            reward_threshold,
            reward_sensitivity,
            reward_window_size,
        )
    else:
        losses, rewards_n, memories = run_rainbow_parallel(
            rom_path,
            state_path,
            episode_length,
            device,
            num_episodes,
            batch_size,
            sight,
            runs_per_worker,
            num_workers,
            memories,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            beta_start,
            beta_frames,
            gamma,
            update_target_every,
            policy_net,
            target_net,
            optimizer,
            replay_buffer,
            losses,
            rewards,
            checkpoint_interval,
            checkpoint_path,
            epsilon_by_location,
            frames_in_loc,
            extra_files,
            reward_locations_xy,
        )

    total_time = time.time() - start_time  # Total training time

    # Given we know frames in location, we can calculate the exact epsilon value
    # for each location
    for loc in frames_in_loc:
        epsilons_by_location[loc] = epsilon_by_frame(
            frames_in_loc[loc], epsilon_start, epsilon_final, epsilon_decay
        )

    # Prepare logging data
    log_data = {
        "total_time": total_time,
        "average_reward": sum(rewards) / len(rewards),
        "losses": losses,
        "epsilon_values": epsilon_values,
        "beta_values": beta_values,
        "td_errors": td_errors,
        "frames_in_loc": frames_in_loc,
        "epsilons_by_location": epsilons_by_location,
    }

    # Save logged data to file
    # check folder exists and create
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    with open("./logs/training_log.json", "w") as outfile:
        json.dump(log_data, outfile, indent=4)
    print("Training log saved to ./logs/training_log.json")

    # Plot results
    plot_best_attempts(
        "./results/", num_episodes, f"RainbowDQN_{run_parallel}_final", rewards
    )

    # Save checkpoint
    save_checkpoint(
        {
            "episode": num_episodes + start_episode,
            "frame_idx": frame_idx,
            "policy_net_state_dict": policy_net.state_dict(),
            "target_net_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_buffer": replay_buffer.state_dict(),
            "frames_in_loc": frames_in_loc,
            "rewards": rewards,
            "epsilon_by_location": epsilons_by_location,
        },
        filename=checkpoint_path,
    )

    # run a final episode in eval mode
    print("Running final episode in eval mode")
    env = Controller(
        rom_path,
        state_path,
        timeout=episode_length,
        log_path="./logs/rainbow_env_eval.json",
        use_sight=sight,
        extra_files=extra_files,
    )
    _ = run_single(
        num_episodes + start_episode,
        2,
        env,
        device,
        policy_net,
        target_net,
        optimizer,
        replay_buffer,
        checkpoint_path,
        frame_idx,
        epsilon_start,
        epsilon_final,
        epsilon_decay,
        beta_start,
        beta_frames,
        batch_size,
        gamma,
        update_target_every,
        losses,
        epsilon_values,
        beta_values,
        td_errors,
        rewards,
        checkpoint_interval,
        epsilon_by_location,
        frames_in_loc,
        reward_threshold,
        reward_sensitivity,
        reward_window_size,
        eval_mode=True,
    )