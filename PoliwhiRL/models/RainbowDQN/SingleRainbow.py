# -*- coding: utf-8 -*-
from PoliwhiRL.models.RainbowDQN.utils import (
    compute_td_error,
    optimize_model,
    save_checkpoint,
)
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts
from tqdm import tqdm
from PoliwhiRL.models.RainbowDQN.utils import beta_by_frame, epsilon_by_frame_cyclic
import random
import torch


def run(
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
    frames_in_loc
):

    for episode in tqdm(range(start_episode, start_episode + num_episodes)):
        state = env.reset()
        state = image_to_tensor(state, device)

        total_reward = 0
        ep_len = 0
        while True:
            frame_idx += 1
            frames_in_loc[env.get_current_location()] += 1
            epsilon = epsilon_by_frame_cyclic(
                frames_in_loc[env.get_current_location()]
                if epsilon_by_location
                else frame_idx,
                epsilon_start,
                epsilon_final,
                epsilon_decay,
            )
            epsilon_values.append(epsilon)  # Log epsilon value
            beta = beta_by_frame(
                frames_in_loc[env.get_current_location()]
                if epsilon_by_location
                else frame_idx,
                beta_start,
                beta_frames,
            )
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
            if done:
                break
            ep_len += 1
        rewards.append(total_reward)
        if episode % 100 == 0 and episode > 0:
            plot_best_attempts("./results/", "", "RainbowDQN_latest_single", rewards)
        if episode % checkpoint_interval == 0:
            save_checkpoint(
                {
                    "episode": num_episodes + start_episode,
                    "frame_idx": frame_idx,
                    "policy_net_state_dict": policy_net.state_dict(),
                    "target_net_state_dict": target_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "replay_buffer": replay_buffer.state_dict(),
                    "frames_in_loc": frames_in_loc
                },
                filename=f"{checkpoint_path.replace('.pth','')}_ep{episode}.pth",
            )

    env.close()

    return losses, rewards, frame_idx
