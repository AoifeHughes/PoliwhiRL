import os
import torch.optim as optim
import torch
import numpy as np
import random
import time
import json
from tqdm import tqdm
from PoliwhiRL.environment.controls import Controller
from PoliwhiRL.models.RainbowDQN.RainbowDQN import RainbowDQN
from PoliwhiRL.models.RainbowDQN.ReplayBuffer import PrioritizedReplayBuffer
from PoliwhiRL.models.RainbowDQN.utils import compute_td_error, optimize_model, save_checkpoint, load_checkpoint
from PoliwhiRL.models.RainbowDQN.utils import beta_by_frame, epsilon_by_frame
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts


def run(
    rom_path,
    state_path,
    episode_length,
    device,
    num_episodes,
    batch_size,
    checkpoint_path="rainbow_checkpoint.pth",
):
    start_time = time.time()  # For computational efficiency tracking
    env = Controller(
        rom_path, state_path, timeout=episode_length, log_path="./logs/rainbow_env.json"
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

    for episode in tqdm(range(start_episode, start_episode + num_episodes)):
        state = env.reset()
        state = image_to_tensor(state, device)

        total_reward = 0
        ep_len = 0
        while True:
            frame_idx += 1
            #frame_loc_idx = env.get_frames_in_current_location()
            epsilon = epsilon_by_frame(
                frame_idx, epsilon_start, epsilon_final, epsilon_decay
            )
            epsilon_values.append(epsilon)  # Log epsilon value
            beta = beta_by_frame(frame_idx, beta_start, beta_frames)
            beta_values.append(beta)  # Log beta value

            if random.random() > epsilon:
                with torch.no_grad():
                    state_t = state.unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    action = q_values.max(1)[1].item()
            else:
                action = env.random_move()

            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, device)
            action = torch.tensor([action], device=device)
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)
            td_error = compute_td_error(
                (state, action, reward, next_state, done),
                policy_net,
                target_net,
                device,
                gamma,
            )
            td_errors.append(td_error)  # Log TD error
            replay_buffer.add(state, action, reward, next_state, done, error=td_error)
            state = next_state
            total_reward += reward.item()

            loss = optimize_model(
                beta,
                policy_net,
                target_net,
                replay_buffer,
                optimizer,
                device,
                batch_size,
                gamma,
            )
            if loss is not None:
                losses.append(loss)

            if frame_idx % update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if episode % 100 == 0 or episode == num_episodes - 1:
                env.record(episode, 1, "Rainbow")
            if done:
                break
            ep_len += 1
        rewards.append(total_reward)
        if episode % 100 == 0 and episode > 0:
            plot_best_attempts("./results/", '', f"Rainbow DQN_latest", rewards)


    total_time = time.time() - start_time  # Total training time
    env.close()

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
            "episode": episode,
            "frame_idx": frame_idx,
            "policy_net_state_dict": policy_net.state_dict(),
            "target_net_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_buffer": replay_buffer.state_dict()
        },
        filename=checkpoint_path,
    )