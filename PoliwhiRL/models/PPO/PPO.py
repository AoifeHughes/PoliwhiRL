# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from PoliwhiRL.models.RainbowDQN.rainbowDQN import Attention

class FeatureCNN(nn.Module):
    def __init__(self, input_dim):
        super(FeatureCNN, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Flatten(),
        )
        self.attention = Attention(self.feature_size(input_dim), 256)

    def feature_size(self, input_dim):
        return self.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1)

    def forward(self, x):
        return self.attention(self.feature_layer(x))

class Actor(nn.Module):
    def __init__(self, feature_dim, num_actions):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=512, batch_first=True)
        self.policy_head = nn.Linear(512, num_actions)

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        return F.softmax(self.policy_head(lstm_out[:, -1, :]), dim=-1)


class Critic(nn.Module):
    def __init__(self, feature_dim):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=512, batch_first=True)
        self.value_head = nn.Linear(512, 1)

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        return self.value_head(lstm_out[:, -1, :])


class PPOModel(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(PPOModel, self).__init__()
        self.FeatureCNN = FeatureCNN(input_dim)
        self.actor = Actor(
            self.FeatureCNN.feature_layer(torch.zeros(1, *input_dim))
            .view(1, -1)
            .size(1),
            num_actions,
        )
        self.critic = Critic(
            self.FeatureCNN.feature_layer(torch.zeros(1, *input_dim))
            .view(1, -1)
            .size(1)
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height, width)
        shared_features = self.FeatureCNN(x)
        shared_features = shared_features.view(batch_size, seq_len, -1)
        action_probs = self.actor(shared_features)
        value_estimates = self.critic(shared_features)
        return action_probs, value_estimates
