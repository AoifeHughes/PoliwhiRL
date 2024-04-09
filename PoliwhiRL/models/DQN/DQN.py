import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_out_size(state_size)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, action_size)

    def _get_conv_out_size(self, state_size):
        x = torch.zeros(1, *state_size)
        x = self.conv1(x)
        x = self.conv2(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size, gamma, lr, epsilon, epsilon_decay, epsilon_min, memory_size, device, target_update=200):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = ReplayMemory(memory_size)
        self.device = device
        self.model = DQNModel(state_size, action_size).to(self.device)
        self.target_model = DQNModel(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.target_update = target_update
        self.update_counter = 0

    def memorize(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(np.transpose(state, (2, 0, 1)), dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(np.transpose(next_state, (2, 0, 1)), dtype=torch.float32).to(self.device)
        self.memory.add(state_tensor, action, reward, next_state_tensor, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)      
        state = torch.tensor(np.transpose(state, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if len(self.memory) < self.batch_size:
            return

        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*transitions)
        batch_states = torch.stack(batch_states)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(self.device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(self.device)
        batch_next_states = torch.stack(batch_next_states)
        batch_dones = torch.tensor(batch_dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        current_q_values = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(batch_next_states).max(1)[0]
        expected_q_values = batch_rewards + (1 - batch_dones) * self.gamma * next_q_values

        td_errors = (current_q_values - expected_q_values.detach()).abs().cpu().detach().numpy()
        self.memory.update_priorities(indices, td_errors)

        self.optimizer.zero_grad()
        loss = (weights * (current_q_values - expected_q_values.detach()) ** 2).mean()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


class ReplayMemory:
    def __init__(self, memory_size, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.priorities = deque(maxlen=memory_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

    def add(self, state, action, reward, next_state, done):
        # Normalize the state and next_state
        state = state / 255.0
        next_state = next_state / 255.0

        max_priority = max(self.priorities) if self.memory else 1.0
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[i] for i in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = (priority + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.memory)


class EpisodicMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.current_episode = []

    def add(self, state, action, reward, next_state, done):
        self.current_episode.append((state, action, reward, next_state, done))
        if done:
            self.memory.append(self.current_episode)
            self.current_episode = []

    def sample(self, batch_size):
        episodes = []
        if len(self.memory) >= batch_size:
            episodes = random.sample(self.memory, batch_size)
        else:
            episodes = list(self.memory)
        return episodes

    def __len__(self):
        return len(self.memory)