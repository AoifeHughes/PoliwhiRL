# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
from .utils import (
    optimize_model,
    epsilon_by_frame,
    store_experience,
    beta_by_frame,
    select_action_hybrid,
)
from PoliwhiRL.utils import image_to_tensor


def run(config, env, policy_net, target_net, optimizer, replay_buffer):
    rewards, losses, epsilon_values, beta_values, td_errors = [], [], [], [], []

    num_actions = len(env.action_space)
    episodes = config.get("start_episode", 0)
    frame_idx = config.get("frame_idx", 0)


    for episode in tqdm(range(episodes, episodes + config["num_episodes"])):
        policy_net.reset_noise()
        episode_reward = 0
        done = False
        state = env.reset()
        state = image_to_tensor(state, config["device"])
        while not done:
            epsilon = epsilon_by_frame(frame_idx, config["epsilon_start"], config["epsilon_final"], config["epsilon_decay"])
            epsilon_values.append(epsilon)

            action = 2 # testing
            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, config["device"])
            beta = beta_by_frame(frame_idx, config["beta_start"], config["beta_frames"])
            beta_values.append(beta)
            td_error = store_experience(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    policy_net,
                    target_net,
                    replay_buffer,
                    config,

                )
            td_errors.append(td_error)

            if len(replay_buffer) >= config["batch_size"]:

                loss = optimize_model(
                    beta,
                    policy_net,
                    target_net,
                    replay_buffer,
                    optimizer,
                    config["device"],
                    config["batch_size"],
                    config["gamma"]
                )
                if loss is not None:
                    losses.append(loss)

            if frame_idx % config["target_update"] == 0:
                target_net.load_state_dict(policy_net.state_dict())

            episode_reward += reward
            frame_idx += 1

        rewards.append(episode_reward)
        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")

    return rewards, losses, epsilon_values, beta_values, td_errors