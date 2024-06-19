import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from PoliwhiRL.environment.controller import Controller
from .rnn import RNN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class PPO:
    def __init__(self, model, lr, gamma, clip_epsilon, ppo_epochs, batch_size):
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
    def update(self, states, actions, rewards, dones, next_states):
        states = torch.stack(states).float().to(device)
        actions = torch.tensor(actions).long().to(device)
        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)
        next_states = torch.stack(next_states).float().to(device)

        # Compute discounted rewards-to-go
        discounted_rewards = compute_discounted_rewards(rewards, self.gamma).to(device)  # Ensure tensor is on correct device
        
        for _ in range(self.ppo_epochs):
            hidden = self.model.init_hidden(self.batch_size)
            
            logits, values, _ = self.model(states, hidden)
            probs = nn.functional.softmax(logits, dim=-1)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_logits, next_values, _ = self.model(next_states, hidden)
                next_probs = nn.functional.softmax(next_logits, dim=-1)
                next_log_probs = nn.functional.log_softmax(next_logits, dim=-1)
                
                next_action = torch.argmax(next_probs, dim=1)
                next_action_log_probs = next_log_probs.gather(1, next_action.unsqueeze(1)).squeeze(1)
                
                target_values = discounted_rewards
            
            advantages = target_values - values.squeeze()
            ratios = torch.exp(action_log_probs - action_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values.squeeze(), target_values)
            loss = policy_loss + 0.5 * value_loss - 0.01 * probs.entropy().mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def compute_discounted_rewards(rewards, gamma):
    rewards = rewards.cpu().numpy()  # Ensure rewards tensor is on CPU before converting to NumPy array
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return torch.tensor(discounted_rewards).float()  # Convert back to tensor


def run(**config):
    # Hyperparameters
    lr = 0.0003
    gamma = 0.99
    clip_epsilon = 0.2
    ppo_epochs = 4
    batch_size = 64
    num_episodes = 1000
    env = Controller(config)
    
    # Initialize the model and PPO
    model = RNN(len(env.action_space)).to(device)
    ppo = PPO(model, lr, gamma, clip_epsilon, ppo_epochs, batch_size)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        
        while not done:
            states = []
            actions = []
            rewards = []
            dones = []
            next_states = []
            
            for _ in range(batch_size):
                hidden = model.init_hidden(1)
                state_tensor = torch.from_numpy(np.transpose(state, (2, 0, 1))).float().unsqueeze(0).to(device)
                logits, _, _ = model(state_tensor.unsqueeze(0), hidden) # Adding batch dimension to state_tensor for processing
                probs = nn.functional.softmax(logits, dim=-1)
                action_dist = Categorical(probs)
                action = action_dist.sample()
                
                next_state, reward, done = env.step(action.item())
                
                states.append(state_tensor.squeeze(0))
                actions.append(action.item())
                rewards.append(reward)
                dones.append(done)
                next_states.append(torch.from_numpy(np.transpose(next_state, (2, 0, 1))).float())
                
                state = next_state
                episode_rewards.append(reward)
                
                if done:
                    break
            
            ppo.update(states, actions, rewards, dones, next_states)
        
        print(f"Episode {episode+1}: Total Reward = {sum(episode_rewards)}")
