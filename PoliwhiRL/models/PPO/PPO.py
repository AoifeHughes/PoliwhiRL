# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PoliwhiRL.models.RainbowDQN.rainbowDQN import Attention


class FeatureCNN(nn.Module):
    def __init__(self, input_dim):
        super(FeatureCNN, self).__init__()

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
            nn.Flatten(),
        )
        self.attention = Attention(self.feature_size(input_dim), 256)

    def feature_size(self, input_dim):
        return self.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1)

    def forward(self, x):
        return self.attention(self.feature_layer(x))

class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim, num_layers, num_heads, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Actor(nn.Module):
    def __init__(self, feature_dim, num_actions, num_transformer_layers=1, num_heads=4, dim_feedforward=512):
        super(Actor, self).__init__()
        self.transformer_encoder = TransformerEncoder(feature_dim, num_transformer_layers, num_heads, dim_feedforward)
        self.policy_head = nn.Linear(feature_dim, num_actions)

    def forward(self, src):
        transformer_out = self.transformer_encoder(src)
        return F.softmax(self.policy_head(transformer_out[-1]), dim=-1)

class Critic(nn.Module):
    def __init__(self, feature_dim, num_transformer_layers=1, num_heads=4, dim_feedforward=512):
        super(Critic, self).__init__()
        self.transformer_encoder = TransformerEncoder(feature_dim, num_transformer_layers, num_heads, dim_feedforward)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, src):
        transformer_out = self.transformer_encoder(src)
        return self.value_head(transformer_out[-1])

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
