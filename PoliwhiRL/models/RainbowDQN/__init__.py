# -*- coding: utf-8 -*-
import os
import torch.optim as optim
import time
import json
from PoliwhiRL.environment.controls import Controller
from PoliwhiRL.models.RainbowDQN.RainbowDQN import RainbowDQN
from PoliwhiRL.models.RainbowDQN.ReplayBuffer import PrioritizedReplayBuffer
from PoliwhiRL.models.RainbowDQN.utils import (
    save_checkpoint,
    load_checkpoint,
)
from PoliwhiRL.models.RainbowDQN.SingleRainbow import run as run_single
from PoliwhiRL.models.RainbowDQN.DoubleRainbow import run as run_rainbow_parallel



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
    memories=0
):
    start_time = time.time()  # For computational efficiency tracking
    env = Controller(
        rom_path, state_path, timeout=episode_length, log_path="./logs/rainbow_env.json", use_sight=sight
    )
    gamma = 0.99
    alpha = 0.6
    beta_start = 0.4
    beta_frames = 1000
    frame_idx = 0
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000
    learning_rate = 1e-4
    capacity = 10000
    update_target_every = 1000
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

    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint is not None:
        start_episode = checkpoint["episode"] + 1
        frame_idx = checkpoint["frame_idx"]
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        target_net.load_state_dict(checkpoint["target_net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    else:
        start_episode = 0


    if not run_parallel:
        losses, rewards, memories = run_single(
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
            start_time,
        )
    else:
       losses, rewards, memories = run_rainbow_parallel(
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
            rewards
        )
    total_time = time.time() - start_time  # Total training time
     # Prepare logging data
    log_data = {
        "total_time": total_time,
        "average_reward": sum(rewards) / len(rewards),
        "losses": losses,
        "epsilon_values": epsilon_values,
        "beta_values": beta_values,
        "td_errors": td_errors,
    }

    # Save logged data to file
    # check folder exists and create
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    with open("./logs/training_log.json", "w") as outfile:
        json.dump(log_data, outfile, indent=4)
    print("Training log saved to ./logs/training_log.json")

    # Save checkpoint
    save_checkpoint(
        {
            "episode": num_episodes+start_episode,
            "frame_idx": frame_idx,
            "policy_net_state_dict": policy_net.state_dict(),
            "target_net_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_buffer": replay_buffer.state_dict(),
        },
        filename=checkpoint_path,
    )



