import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_out_size(state_size)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu6 = nn.ReLU()
        
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc3 = nn.Linear(128, action_size)

    def _get_conv_out_size(self, state_size):
        x = torch.zeros(1, *state_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x, _ = self.lstm(x.unsqueeze(1))
        x = self.fc3(x.squeeze(1))
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size, gamma, lr, epsilon, epsilon_decay, epsilon_min, memory_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = EpisodicMemory(memory_size)
        self.device = device
        self.model = DQNModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

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

        episodes = self.memory.sample(self.batch_size)

        for episode in episodes:
            states, actions, rewards, next_states, dones = zip(*episode)
            states = torch.stack(states)
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

            q_values = self.model(states)
            next_q_values = self.model(next_states)
            targets = q_values.clone()

            for i in range(len(episode)):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, targets)
            loss.backward()
            self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

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