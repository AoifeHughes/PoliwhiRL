import multiprocessing as mp
import torch
import torch.optim as optim
import os
import random
import numpy as np
import time
from PoliwhiRL.models.RainbowDQN.RainbowDQN import RainbowDQN
from PoliwhiRL.models.RainbowDQN.ReplayBuffer import PrioritizedReplayBuffer
from PoliwhiRL.models.RainbowDQN.utils import compute_td_error, optimize_model, beta_by_frame, epsilon_by_frame, load_checkpoint, save_checkpoint
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts, save_results
from PoliwhiRL.environment.controls import Controller



def worker(worker_id, env_fn, epsilon_start, epsilon_final, epsilon_decay, beta_start, beta_frames, policy_net, target_net, device, num_episodes, experience_queue, td_error_queue, reward_queue, frame_idx_queue):
    local_env = env_fn()  # Create a local instance of the environment
    frame_idx = 0
    for episode in range(num_episodes):
        state = local_env.reset()
        state = image_to_tensor(state, device)
        total_reward = 0
        episode_td_errors = []
        episode_experiences = []
        frame_idxs = []
        done = False
        while not done:
            frame_idx += 1
            frame_idxs.append(frame_idx)
            epsilon = epsilon_by_frame(frame_idx, epsilon_start, epsilon_final, epsilon_decay)
            beta = beta_by_frame(frame_idx, beta_start, beta_frames)  # This could be passed back if needed
            state_t = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

            if random.random() > epsilon:
                with torch.no_grad():
                    q_values = policy_net(state_t)
                    action = q_values.max(1)[1].item()
            else:
                action = local_env.random_move()

            next_state, reward, done, _ = local_env.step(action)
            next_state = image_to_tensor(next_state, device)
            total_reward += reward

            with torch.no_grad():
                td_error = compute_td_error((state, action, reward, next_state, done), policy_net, target_net, device)
                episode_td_errors.append(td_error)

            episode_experiences.append((state, action, reward, next_state, done, beta))  # Include beta if needed
            state = next_state
        
        # After episode ends, put all experiences and metrics in their respective queues
        experience_queue.put(episode_experiences)
        td_error_queue.put(episode_td_errors) 
        reward_queue.put(total_reward)  
        frame_idx_queue.put(frame_idx) 


def aggregate_and_update_model(experience_queue, policy_net, target_net, optimizer, replay_buffer, device, batch_size, gamma, update_target_every, beta_start, beta_frames, frame_idx):
    memories_processed = 0
    while not experience_queue.empty():
        experiences = experience_queue.get()
        for experience in experiences:
            state, action, reward, next_state, done = experience
            state, next_state = torch.tensor(state, device=device), torch.tensor(next_state, device=device)
            action, reward, done = torch.tensor([action], device=device), torch.tensor([reward], device=device), torch.tensor([done], device=device)
            td_error = compute_td_error((state, action, reward, next_state, done), policy_net, target_net, device, gamma)
            replay_buffer.add(state, action, reward, next_state, done, td_error)
            memories_processed += 1
            # Check if it's time to update the target network
            if memories_processed % update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())
    
    # Assume we're now passing beta directly to this function
    if len(replay_buffer) >= batch_size:
        beta = beta_by_frame(frame_idx, beta_start, beta_frames)  # frame_idx needs to be tracked and passed
        loss = optimize_model(beta, policy_net, target_net, optimizer, replay_buffer, device, batch_size, gamma)
        return loss
    return None


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
    
    # Initialize the environment function here if needed
    def env_fn():
        return Controller(rom_path, state_path, timeout=episode_length, log_path="./logs/rainbow_env.json")
    env = env_fn()


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

    num_workers = 4
    experience_queue = mp.Queue()
    td_error_queue = mp.Queue()
    reward_queue = mp.Queue()
    frame_idx_queue = mp.Queue()
    processes = []

    # Define a worker wrapper to handle environments and queues
    def worker_wrapper(worker_id, num_episodes, experience_queue):
        worker(worker_id, env_fn, epsilon_start, epsilon_final, epsilon_decay, beta_start, beta_frames, policy_net, target_net, device, num_episodes, experience_queue, td_error_queue, reward_queue, frame_idx_queue)

    mp.set_start_method('spawn')
    for i in range(num_workers):
        num_episodes_per_worker = num_episodes // num_workers
        p = mp.Process(target=worker_wrapper, args=(i, num_episodes_per_worker, experience_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    experiences = []
    while not experience_queue.empty():
        experiences.extend(experience_queue.get())
    
    td_errors = []
    while not td_error_queue.empty():
        td_errors.extend(td_error_queue.get())
    
    rewards = []
    while not reward_queue.empty():
        rewards.append(reward_queue.get())
    
    frame_idxs = []
    while not frame_idx_queue.empty():
        frame_idxs.append(frame_idx_queue.get())


    loss = None
    if len(experiences) > 0:
        loss = aggregate_and_update_model(experiences, policy_net, target_net, optimizer, replay_buffer, device, batch_size, gamma, update_target_every, beta_start, beta_frames, frame_idxs)
        if loss is not None:
            losses.append(loss)

        print(f"Model updated. Loss: {loss}")

    total_time = time.time() - start_time  # Total training time
    print(f"Total training time: {total_time} seconds")

    # Save your model and other important data here
    save_checkpoint(
    {
        "episode": num_episodes,
        "frame_idx": frame_idx,
        "policy_net_state_dict": policy_net.state_dict(),
        "target_net_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_buffer": replay_buffer.state_dict(),  # Assuming replay_buffer has a method to return its state
        # Add any other parameters you need
    },
    filename=checkpoint_path,
    )
