# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .noisylinear import NoisyLinear


class Attention(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x):
        attention_scores = self.attention_layer(x)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_features = x * attention_weights
        return weighted_features.sum(1)


class RainbowDQN(nn.Module):
    def __init__(self, input_dim, num_actions, device, atom_size=51, Vmin=-10, Vmax=10):
        super(RainbowDQN, self).__init__()
        self.num_actions = num_actions
        self.atom_size = atom_size
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.support = (
            torch.linspace(Vmin, Vmax, atom_size).view(1, 1, atom_size).to(device)
        )
        # Define network layers
        self.feature_layer = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_input_dim = self.feature_size(input_dim)
        #self.attention = Attention(self.fc_input_dim, 256)
        self.lstm = nn.LSTM(
            input_size=self.fc_input_dim, hidden_size=512, batch_first=True
        )
        self.value_stream = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, atom_size)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * atom_size),
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        # Reshape x to treat the sequence as a batch dimension
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.feature_layer(x)
        # Calculate feature dimension for reshaping
        x = x.view(batch_size, seq_len, -1)  # Reshape to [batch_size, seq_len, features]
        # No need to reshape to [batch_size, 1, -1], LSTM can handle [batch_size, seq_len, features]
        lstm_out, _ = self.lstm(x)
        # Select the last output for each sequence
        x = lstm_out[:, -1, :]
        dist = self.get_distribution(x)
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values

    def get_distribution(self, x):
        value = self.value_stream(x).view(-1, 1, self.atom_size)
        advantage = self.advantage_stream(x).view(-1, self.num_actions, self.atom_size)
        advantage_mean = advantage.mean(1, keepdim=True)
        dist = value + advantage - advantage_mean
        dist = F.softmax(dist, dim=-1)
        dist = dist.clamp(min=1e-3)  # Avoid zeros
        return dist

    def feature_size(self, input_dim):
        return self.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1)

    def reset_noise(self):
        """Reset all noisy layers"""
        for _, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
