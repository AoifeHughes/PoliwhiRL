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

        # Define the attention layer
        self.attention = Attention(self.fc_input_dim, 256)

        # Modify LSTM input size to match the flattened feature dimension
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
        # Determine the shape and adjust dimensions accordingly
        original_shape = x.size()
        unbatched = False
        if len(original_shape) == 3:  # Unbatched single image [C, H, W]
            x = x.unsqueeze(0).unsqueeze(0)  # Reshape to [1, 1, C, H, W]
            unbatched = True
        elif len(original_shape) == 4:  # Unbatched sequence [S, C, H, W]
            x = x.unsqueeze(0)  # Reshape to [1, S, C, H, W]
            unbatched = True

        batch_size, sequence_length, _, _, _ = x.size()

        # Flatten the sequence to [B*S, C, H, W] for CNN processing
        x = x.view(batch_size * sequence_length, original_shape[-3], original_shape[-2], original_shape[-1])

        # Process through convolutional and attention layers
        x = self.feature_layer(x)

        # After CNN, reshape back to [B, S, -1] for attention and LSTM
        x = x.view(batch_size, sequence_length, -1)
        x = self.attention(x)

        # Correctly processing through LSTM
        lstm_out, _ = self.lstm(x)
        # Select the last output for further processing, ensuring it's properly handled for unbatched inputs
        if unbatched:
            x = lstm_out.squeeze(0)  # For unbatched inputs, remove the batch dimension
        else:
            x = lstm_out[:, -1, :]
        
        # Continue with distribution and Q value calculations
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
