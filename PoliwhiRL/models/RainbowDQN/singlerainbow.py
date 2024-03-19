# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
from PoliwhiRL.models.RainbowDQN.evaluate import evaluate_model
from PoliwhiRL.models.RainbowDQN.utils import (
    optimize_model,
    save_checkpoint,
    epsilon_by_frame,
    store_experience,
    beta_by_frame,
    select_action_hybrid,
)
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts, plot_losses, weighted_random_indices


def run(config, env, policy_net, target_net, optimizer, replay_buffer):
    rewards, losses, epsilon_values, beta_values, td_errors, eval_rewards, buttons = [], [], [], [], [], [], []

    num_actions = len(env.action_space)
    action_counts = np.zeros(num_actions)
    action_rewards = np.zeros(num_actions)
    episodes = config.get("start_episode", 0)
    frame_idx = config.get("frame_idx", 0)
    sequence_length = config.get("sequence_length", 4)  # Assuming a fixed sequence length

    for episode in tqdm(range(episodes, episodes + config["num_episodes"])):
        policy_net.reset_noise()
        state_sequence = []
        action_sequence = []
        reward_sequence = []
        next_state_sequence = []
        done_sequence = []
        total_reward = 0
        done = False
        state = env.reset()
        state = image_to_tensor(state, config["device"])

        while not done and len(state_sequence) < sequence_length:
            epsilon = epsilon_by_frame(frame_idx, config["epsilon_start"], config["epsilon_final"], config["epsilon_decay"])
            epsilon_values.append(epsilon)

            action, was_random = select_action_hybrid(
                    state,
                    policy_net,
                    config,
                    frame_idx,
                    action_counts,
                    num_actions,
                    epsilon,
                )
            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, config["device"])

            # Collect experiences until we have a complete sequence
            state_sequence.append(state)
            action_sequence.append(action)
            reward_sequence.append(reward)
            next_state_sequence.append(next_state)
            done_sequence.append(done)

            state = next_state
            total_reward += reward
            frame_idx += 1

            # When a sequence is complete, store it and reset sequences
            if len(state_sequence) == sequence_length:
                store_experience_sequence(
                    state_sequence,
                    action_sequence,
                    reward_sequence,
                    next_state_sequence,
                    done_sequence,
                    policy_net,
                    target_net,
                    replay_buffer,
                    config,
                    td_errors
                )
                state_sequence.pop(0)
                action_sequence.pop(0)
                reward_sequence.pop(0)
                next_state_sequence.pop(0)
                done_sequence.pop(0)

        # Optimize model after each episode or more frequently based on your configuration
        for _ in range(config['opt_runs']):
            loss = optimize_model_sequence(
                beta_values[-1],  # Latest beta value
                policy_net,
                target_net,
                replay_buffer,
                optimizer,
                config["device"],
                config["batch_size"],
                config["gamma"],
            )
            if loss is not None:
                losses.append(loss)

        # Update target network
        if episode % config["target_update"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards.append(total_reward)

    return losses, epsilon_values, beta_values, td_errors, rewards