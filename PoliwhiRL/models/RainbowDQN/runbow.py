# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
from PoliwhiRL.models.RainbowDQN.evaluate import evaluate_model
from PoliwhiRL.models.RainbowDQN.training_functions import (
    optimize_model_sequence,
    save_checkpoint,
    epsilon_by_frame,
    store_experience_sequence,
    beta_by_frame,
    select_action_hybrid,
    populate_replay_buffer,
)
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts, plot_losses


def run(config, env, policy_net, target_net, optimizer, replay_buffer):
    rewards, losses, epsilon_values, beta_values, td_errors, eval_rewards, buttons = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    num_actions = len(env.action_space)
    action_counts = np.zeros(num_actions)
    action_rewards = np.zeros(num_actions)
    episodes = config.get("start_episode", 0)
    frame_idx = config.get("frame_idx", 0)

    print("\nPopulating replay buffer...\n")
    while True:
        populate_replay_buffer(
            config, env, replay_buffer, policy_net, target_net, td_errors
        )
        if len(replay_buffer) >= config["capacity"]:
            break
    print(
        f"\n Number of memories stored: {len(replay_buffer)} / {config['capacity']}\n"
    )
    env.run = 1

    print("\nTraining...\n")
    for episode in (pbar := tqdm(range(episodes, episodes + config["num_episodes"]))):
        frame_idx = run_episode(
            config,
            env,
            buttons,
            policy_net,
            target_net,
            replay_buffer,
            rewards,
            frame_idx,
            epsilon_values,
            beta_values,
            action_counts,
            action_rewards,
            td_errors,
        )
        model_updates(
            episode,
            config,
            policy_net,
            target_net,
            replay_buffer,
            optimizer,
            beta_values,
            losses,
        )
        post_run_jobs(
            episode,
            config,
            env,
            policy_net,
            target_net,
            replay_buffer,
            optimizer,
            eval_rewards,
            rewards,
            losses,
            frame_idx,
        )
        pbar.set_description(
            f"Episode {episode+1} | Last reward: {rewards[-1]:.2f} | Avg reward: {np.mean(rewards[-100:]):.2f} | Best reward: {np.max(rewards):.2f}"
        )
    return losses, beta_values, td_errors, rewards


def run_episode(
    config,
    env,
    buttons,
    policy_net,
    target_net,
    replay_buffer,
    rewards,
    frame_idx,
    epsilon_values,
    beta_values,
    action_counts,
    action_rewards,
    td_errors,
):

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
    sequence_length = config.get("sequence_length", 4)
    num_actions = len(env.action_space)

    while not done:
        epsilon = epsilon_by_frame(
            frame_idx,
            config["epsilon_start"],
            config["epsilon_final"],
            config["epsilon_decay"],
        )
        epsilon_values.append(epsilon)
        beta = beta_by_frame(frame_idx, config["beta_start"], config["beta_frames"])
        beta_values.append(beta)

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
        action_rewards[action] += reward
        env.record(epsilon, "rdqn", was_random, 0)

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
                td_errors,
            )
            state_sequence = []
            action_sequence = []
            reward_sequence = []
            next_state_sequence = []
            done_sequence = []
    rewards.append(total_reward)
    buttons.append(env.buttons)
    return frame_idx


def model_updates(
    episode,
    config,
    policy_net,
    target_net,
    replay_buffer,
    optimizer,
    beta_values,
    losses,
):
    for _ in range(config["opt_runs"]):
        loss = optimize_model_sequence(
            beta_values[-1],
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


def document(eval_rewards, rewards, losses):
    plot_best_attempts("./results/", 0, "RainbowDQN_latest_single_eval", eval_rewards)
    plot_best_attempts("./results/", 0, "RainbowDQN_latest_single", rewards)
    plot_losses("./results/", 0, losses)


def post_run_jobs(
    episode,
    config,
    env,
    policy_net,
    target_net,
    replay_buffer,
    optimizer,
    eval_rewards,
    rewards,
    losses,
    frame_idx,
):
    if episode % config["eval_frequency"] == 0:
        eval_rewards.append(evaluate_model(config, env, policy_net))
        document(eval_rewards, rewards, losses)

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
