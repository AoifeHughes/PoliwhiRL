import random
import torch
from tqdm import tqdm
from PoliwhiRL.models.RainbowDQN.utils import (
    optimize_model, save_checkpoint, epsilon_by_frame, store_experience, beta_by_frame
)
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts

def run(config, env, policy_net, target_net, optimizer, replay_buffer):
    frame_idx = config.get('frame_idx', 0)
    rewards, losses, epsilon_values, beta_values, td_errors = [], [], [], [], []
    frames_in_loc = {i: 0 for i in range(256)}  # Assuming 256 possible locations, adjust as needed

    for episode in tqdm(range(config.get('start_episode', 0), config.get('start_episode', 0) + config['num_episodes'])):
        state = env.reset()
        state = image_to_tensor(state, config['device'])
        total_reward = 0
        done = False

        while not done:
            epsilon = epsilon_by_frame(frame_idx, config['epsilon_start'], config['epsilon_final'], config['epsilon_decay'])
            epsilon_values.append(epsilon)
            action, was_random = select_action(state, epsilon, env, policy_net, config)
            next_state, reward, done = env.step(action)
            
            next_state = image_to_tensor(next_state, config['device'])

            if not config.get('eval_mode', False):
                # Store experience using the dedicated function
                store_experience(state, action, reward, next_state, done, policy_net, target_net, replay_buffer, config, td_errors, frame_idx)
                beta = beta_by_frame(frame_idx, config['beta_start'], config['beta_frames'])
                # Optimize model after storing experience
                loss = optimize_model(beta, policy_net, target_net, replay_buffer, optimizer, config['device'], config['batch_size'], config['gamma'])
                if loss is not None:
                    losses.append(loss)

                if frame_idx % config['target_update'] == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            env.record(epsilon, "rdqn", was_random)
            state = next_state
            total_reward += reward
            frame_idx += 1

        rewards.append(total_reward)
        if episode % config['checkpoint_interval'] == 0 and episode > 0:
            save_checkpoint(config, policy_net, target_net, optimizer, replay_buffer, frames_in_loc, rewards, epsilons_by_location=None)  # epsilons_by_location is not defined in this context

    return losses, rewards, frame_idx



def select_action(state, epsilon, env, policy_net, config):
    if random.random() > epsilon:
        with torch.no_grad():
            q_values = policy_net(state.unsqueeze(0).to(config['device']))
            action = q_values.max(1)[1].view(1, 1).item()
    else:
        action = env.random_move()
    return action, random.random() > epsilon



def update_state(next_state, reward, frame_idx, done):
    """
    Updates the state, total reward, and frame index for the next iteration.
    """
    total_reward += reward
    frame_idx += 1
    return next_state, total_reward, frame_idx, done

def post_episode_processing(episode, total_reward, rewards, config, policy_net, target_net, optimizer, replay_buffer, frames_in_loc, epsilons_by_location):
    """
    Handles logging, checkpointing, and plotting after each episode.
    """
    rewards.append(total_reward)  # Assume rewards is a list tracking rewards per episode

    # Adjust for compatibility with the provided save_checkpoint function
    if episode % config['checkpoint_interval'] == 0 and episode > 0:
        save_checkpoint(
            config=config,
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            frames_in_loc=frames_in_loc,  # Ensure this is tracked throughout the training
            rewards=rewards,
            epsilons_by_location=epsilons_by_location,  # Ensure this is properly managed
            filename=None  # Optional: Specify a filename or leave None to use default from config
        )

    # Optionally, plot best attempts periodically
    if episode % 100 == 0:  # Example interval, adjust as needed
        plot_best_attempts("./results/", "", "RainbowDQN_latest_single", rewards)
