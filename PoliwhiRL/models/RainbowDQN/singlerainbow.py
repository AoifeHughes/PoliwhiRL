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
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts


def run(config, env, policy_net, target_net, optimizer, replay_buffer):
    rewards, losses, epsilon_values, beta_values, td_errors, eval_rewards = [], [], [], [], [], []

    num_actions = len(env.action_space)
    action_counts = np.zeros(num_actions)
    action_rewards = np.zeros(num_actions)
    episodes = config.get("start_episode", 0)
    frame_idx = config.get("frame_idx", 0)

    for episode in (
        pbar := tqdm(
            range(
                episodes,
                episodes + config["num_episodes"],
            )
        )
    ):
        policy_net.reset_noise()
        state = env.reset()
        state = image_to_tensor(state, config["device"])
        total_reward = 0
        done = False

        while not done:
            epsilon = epsilon_by_frame(
                frame_idx,
                config["epsilon_start"],
                config["epsilon_final"],
                config["epsilon_decay"],
            )
            epsilon_values.append(epsilon)

            action, q_value = select_action_hybrid(
                    state,
                    policy_net,
                    config,
                    frame_idx,
                    action_counts,
                    num_actions,
                    epsilon,
                )

            if q_value is None:
                was_random = True
            else:
                was_random = False
            next_state, reward, done = env.step(action)
            next_state = image_to_tensor(next_state, config["device"])
            action_rewards[action] += reward

            beta = beta_by_frame(frame_idx, config["beta_start"], config["beta_frames"])
            beta_values.append(beta)
            priority_val = store_experience(
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
            )
            if config["record"]:
                env.record(epsilon, "rdqn", was_random, priority_val)
            state = next_state
            total_reward += reward
            frame_idx += 1

        # Optimize model after storing experience
        loss = optimize_model(
            beta,
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

        if episode % config["target_update"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards.append(total_reward)
        pbar.set_description(
            f"Episode: {episode}, Frame: {frame_idx}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}, Best reward: {max(rewards):.2f}, Avg reward: {sum(rewards) / len(rewards):.2f}, Frame_idx: {frame_idx}"
        )
        if episode % config["checkpoint_interval"] == 0 and episode > 0:
            save_checkpoint(
                config,
                policy_net,
                target_net,
                optimizer,
                replay_buffer,
                rewards,
                episodes=episode,
                frames=frame_idx,
            )

        if episode % config["plot_interval"] == 0 and episode > 0:
            print("Plotting best attempts...")
            plot_best_attempts("./results/", 0, "RainbowDQN_latest_single", rewards)

        if episode % config['eval_interval'] == 0 and episode > 0:
            print("Evaluating model...")
            avg_eval = evaluate_model(config, env, policy_net)
            eval_rewards.append(avg_eval)

    config.update(
        {
            "start_episode": episodes,
            "frame_idx": frame_idx,
        }
    )
    return losses, beta_values, td_errors, rewards
