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
import numpy as np
import math

from collections import deque

def run(config, env, policy_net, target_net, optimizer, replay_buffer):
    frame_idx = config.get("frame_idx", 0)
    rewards, losses, epsilon_values, td_errors = [], [], [], []
    is_n_step = config.get("n_steps", 1) > 1
    num_actions = len(env.action_space)  # Get the number of actions from the environment
    action_counts = np.zeros(num_actions)
    action_rewards = np.zeros(num_actions)

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

            action, q_val = select_action_ucb(state, policy_net, config, frame_idx, action_counts, action_rewards, num_actions)

            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, config["device"])

            action_rewards[action] += reward

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
            if config["record"]:
                env.record(epsilon, "rdqn")
            state = next_state
            total_reward += reward
            frame_idx += 1

        rewards.append(total_reward)
        pbar.set_description(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon}, Best reward: {max(rewards)}, Avg reward: {sum(rewards)/len(rewards)}")

        if episode % config["checkpoint_interval"] == 0:
            save_checkpoint(config, policy_net, target_net, optimizer, replay_buffer, rewards)
            plot_best_attempts(
                "./results/", episode, "RainbowDQN_latest_single", rewards
            )

    return losses, rewards, frame_idx


def select_action_ucb(state, policy_net, config, frame_idx, action_counts, action_rewards, num_actions):
    exploration_rate = np.sqrt(2 * math.log(frame_idx + 1))

    ucb_values = np.zeros(num_actions)
    for action in range(num_actions):
        if action_counts[action] > 0:
            average_reward = action_rewards[action] / action_counts[action]
            ucb_bonus = exploration_rate / np.sqrt(action_counts[action])
            ucb_values[action] = average_reward + ucb_bonus
        else:
            ucb_values[action] = np.inf

    action = np.argmax(ucb_values)

    with torch.no_grad():
        q_values = policy_net(state.unsqueeze(0).to(config["device"]))
        # This ensures we are selecting a valid action according to the policy network, but guided by UCB
        q_values = q_values.cpu().numpy()
        action_q_value = q_values[0][action]

    action_counts[action] += 1

    return action, action_q_value