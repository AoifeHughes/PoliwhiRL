import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
from PoliwhiRL.models.RainbowDQN.RainbowDQN import RainbowDQN, save_checkpoint, load_checkpoint
from PoliwhiRL.models.RainbowDQN.ReplayBuffer import PrioritizedReplayBuffer
from PoliwhiRL.environment.controls import Controller
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts
from tqdm import tqdm


rom_path = "./emu_files/Pokemon - Crystal Version.gbc"
state_path = "./emu_files/states/start.state"
episode_length = 200  # Max length of each episode
env = Controller(rom_path, state_path, timeout = episode_length)
device = torch.device("mps")
num_episodes = 150  # Total number of episodes to train
batch_size = 32
gamma = 0.99  # Discount factor for future rewards
alpha = 0.6  # Alpha value for PER
beta_start = 0.4  # Initial value for beta
beta_frames = 10000  # Number of frames over which beta will be annealed to 1
frame_idx = 0  # Frame count for beta annealing
epsilon_start = 1.0  # Starting value of epsilon
epsilon_final = 0.01  # Final value of epsilon
epsilon_decay = 30000  # Number of frames over which epsilon will be annealed
learning_rate = 1e-4
capacity = 10000  # Capacity of the replay buffer
update_target_every = 1000  # Update target network every X frames
losses = []  # To store loss values
screen_size = env.screen_size()
input_shape = (3, int(screen_size[0]), int(screen_size[1]))
policy_net = RainbowDQN(input_shape, len(env.action_space), device).to(device)
target_net = RainbowDQN(input_shape, len(env.action_space), device).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set target net to evaluation mode

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = PrioritizedReplayBuffer(capacity, alpha)

def run(checkpoint_path="rainbow_checkpoint.pth.tar"):
    # Load checkpoint if exists
    global frame_idx
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        start_episode = checkpoint['episode'] + 1  # Continue from next episode
        frame_idx = checkpoint['frame_idx']
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Assuming the replay buffer can be saved and loaded correctly
        # This might require custom handling depending on its implementation
        replay_buffer.load_state(checkpoint['replay_buffer'])
        # Load any other parameters you've saved
    else:
        start_episode = 0
    rewards = []    
    for episode in tqdm(range(start_episode, num_episodes)):
        state = env.reset()
        state = image_to_tensor(state, device)
        
        total_reward = 0
        while True:
            frame_idx += 1
            epsilon = epsilon_by_frame(frame_idx)
            beta = beta_by_frame(frame_idx)
            
            # Epsilon-greedy action selection
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
            td_error = compute_td_error((state, action, reward, next_state, done), policy_net, target_net)
            replay_buffer.add(state, action, reward, next_state, done, error=td_error)
            state = next_state
            total_reward += reward.item()  # Assuming reward is a tensor
            
            loss = optimize_model(beta)
            if loss is not None:
                losses.append(loss)
            
            if frame_idx % update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if done:
                break
        
        print(f"Episode: {episode}, Total Reward: {total_reward}, Average Loss: {np.mean(losses) if losses else 0}")
        rewards.append(total_reward)
        
    plot_best_attempts("./rewards/", episode, "Rainbow DQN", rewards)
    # Save checkpoint at the end
    save_checkpoint({
        'episode': episode,
        'frame_idx': frame_idx,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer': replay_buffer.state_dict(),  # Assuming replay_buffer has a method to return its state
        # Add any other parameters you need
    }, filename=checkpoint_path)


def beta_by_frame(frame_idx):
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

def epsilon_by_frame(frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

def compute_td_error(experience, policy_net, target_net, gamma=0.99):
    state, action, reward, next_state, done = experience
    
    # Ensure tensors are on the correct device and add batch dimension since dealing with single experience
    state = state.to(device).unsqueeze(0)  # Add batch dimension
    next_state = next_state.to(device).unsqueeze(0)  # Add batch dimension
    action = torch.tensor([action], device=device, dtype=torch.long)
    reward = torch.tensor([reward], device=device, dtype=torch.float)
    done = torch.tensor([done], device=device, dtype=torch.bool)
    
    # Compute current Q values: Q(s, a)
    current_q_values = policy_net(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
    
    # Compute next Q values from target network
    with torch.no_grad():
        next_state_values = target_net(next_state).max(1)[0].detach()
        next_state_values[done] = 0.0  # Zero-out terminal states
        expected_q_values = reward + gamma * next_state_values

    # TD error
    td_error = (expected_q_values - current_q_values).abs()
    return td_error.item()  # Return absolute TD error as scalar


def optimize_model(beta):
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)
    
    # Directly convert tuples to tensors without np.array conversion
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    rewards = torch.stack(rewards).to(device)
    next_states = torch.stack(next_states).to(device)
    dones = torch.stack(dones).to(device)
    weights = torch.FloatTensor(weights).unsqueeze(-1).to(device)
    
    # Current Q values
    current_q_values = policy_net(states).gather(1, actions)
    
    # Next Q values based on the action chosen by policy_net
    next_q_values = policy_net(next_states).detach()
    _, best_actions = next_q_values.max(1, keepdim=True)
    
    # Next Q values from target_net for actions chosen by policy_net
    next_q_values_target = target_net(next_states).detach().gather(1, best_actions)
    
    # Expected Q values
    expected_q_values = rewards + (gamma * next_q_values_target * (~dones)).float()
    
    # Compute the loss
    loss = (current_q_values - expected_q_values).pow(2) * weights
    prios = loss + 1e-5  # Avoid zero priority
    loss = loss.mean()
    
    # Perform optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update priorities in the buffer
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

    return loss.item()  # Optional: return the loss value for monitoring
