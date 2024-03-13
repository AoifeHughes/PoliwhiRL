# -*- coding: utf-8 -*-
import random
import torch
from tqdm import tqdm
from PoliwhiRL.models.RainbowDQN.utils import (
    optimize_model,
    save_checkpoint,
    epsilon_by_frame,
    store_experience,
    add_n_step_experience,
    beta_by_frame,
)
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts

from collections import deque

def run(config, env, policy_net, target_net, optimizer, replay_buffer):
    frame_idx = config.get("frame_idx", 0)
    rewards, losses, epsilon_values, td_errors = [], [], [], []
    is_n_step = config.get("n_steps", 1) > 1

    # Initialize n-step buffer if needed
    if is_n_step:
        n_step_buffer = deque(maxlen=config['n_steps'])
    else:
        n_step_buffer = None  # Ensure n_step_buffer is None if not used

    for episode in (pbar := tqdm(range(config.get("start_episode", 0), config.get("start_episode", 0) + config["num_episodes"]))):
        policy_net.reset_noise()
        state = env.reset()
        state = image_to_tensor(state, config["device"])
        total_reward = 0
        done = False

        while not done:
            epsilon = epsilon_by_frame(frame_idx, config["epsilon_start"], config["epsilon_final"], config["epsilon_decay"])
            action, was_random = select_action(state, epsilon, env, policy_net, config)
            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, config["device"])

            if not config.get("eval_mode", False):
                # Use store_experience function, passing n_step_buffer when applicable
                beta = beta_by_frame(frame_idx, config["beta_start"], config["beta_frames"])
                store_experience(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    policy_net,
                    target_net,
                    replay_buffer,
                    config,
                    td_errors,
                    beta,
                    n_step_buffer=n_step_buffer  # Pass n_step_buffer if initialized
                )
                loss = optimize_model(beta, policy_net, target_net, replay_buffer, optimizer, config["device"], config["batch_size"], config["gamma"])
                if loss is not None:
                    losses.append(loss)

                if frame_idx % config["target_update"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            state = next_state
            total_reward += reward
            frame_idx += 1

        rewards.append(total_reward)
        pbar.set_description(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon}, Best reward: {max(rewards)}, Avg reward: {sum(rewards)/len(rewards)}")

        if episode % config["checkpoint_interval"] == 0:
            save_checkpoint(config, policy_net, target_net, optimizer, replay_buffer, rewards)

    return losses, rewards, frame_idx

def select_action(state, epsilon, env, policy_net, config):
    was_random = False
    if random.random() > epsilon:
        with torch.no_grad():
            q_values = policy_net(state.unsqueeze(0).to(config["device"]))
            action = q_values.max(1)[1].view(1, 1).item()
    else:
        was_random = True
        action = env.random_move()
    return action, was_random
