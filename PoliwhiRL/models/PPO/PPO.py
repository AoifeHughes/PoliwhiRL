# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: create two models, one for the ram map and one with imgs to save the if statements

class FeatureCNN(nn.Module):
    def __init__(self, input_dim, vision=True):
        super(FeatureCNN, self).__init__()
        self.vision = vision
        if self.vision:
            self.feature_layer = nn.Sequential(
                nn.Conv2d(input_dim[0], 32, kernel_size=3, stride=1, padding=2),
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
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Added an additional convolutional layer
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Flatten(),
            )
        else:
            self.feature_layer = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Flatten(),
            )

    def feature_size(self, input_dim):
        if self.vision:
            return self.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1)
        else:
            return self.feature_layer(torch.zeros(1, input_dim)).view(1, -1).size(1)

    def forward(self, x):
        return self.feature_layer(x)

class Actor(nn.Module):
    def __init__(self, feature_dim, num_actions):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=128, num_layers=2, batch_first=True)  # Increased hidden size and added an extra LSTM layer
        self.policy_head = nn.Sequential(
            nn.Linear(128, 256),  # Added an additional fully connected layer
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        return F.softmax(self.policy_head(lstm_out[:, -1, :]), dim=-1)

class Critic(nn.Module):
    def __init__(self, feature_dim):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=128, num_layers=2, batch_first=True)  # Increased hidden size and added an extra LSTM layer
        self.value_head = nn.Sequential(
            nn.Linear(128, 256),  # Added an additional fully connected layer
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        return self.value_head(lstm_out[:, -1, :])

class PPOModel(nn.Module):
    def __init__(self, input_dim, num_actions, vision=True):
        super(PPOModel, self).__init__()
        self.vision = vision
        self.FeatureCNN = FeatureCNN(input_dim, vision)
        feature_size = self.FeatureCNN.feature_size(input_dim)
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=256, num_layers=2, batch_first=True)
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x, hidden=None):
        if self.vision:
            batch_size, seq_len, channels, height, width = x.size()
            x = x.view(batch_size * seq_len, channels, height, width)
        else:
            batch_size, seq_len, input_dim = x.size()
            x = x.view(batch_size * seq_len, input_dim)

        # Normalize the input images from uint8 to float
        x = x.float() / 255.0

        features = self.FeatureCNN(x)
        features = features.view(batch_size, seq_len, -1)
        if hidden is None:
            lstm_out, hidden = self.lstm(features)
        else:
            lstm_out, hidden = self.lstm(features, hidden)
        
        lstm_out = lstm_out[:, -1, :]  # Use the last output of the LSTM
        
        action_probs = F.softmax(self.actor(lstm_out), dim=-1)
        value_estimates = self.critic(lstm_out)
        
        return action_probs, value_estimates, hidden
