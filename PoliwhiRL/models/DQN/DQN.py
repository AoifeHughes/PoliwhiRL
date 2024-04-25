# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_out_size(state_size)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.relu3 = nn.ReLU()

        self.lstm = nn.LSTM(512, 128)
        self.fc2 = nn.Linear(128, action_size)

    def _get_conv_out_size(self, state_size):
        x = torch.zeros(1, *state_size)
        x = self.conv1(x)
        x = self.conv2(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        # x has shape [batch_size, seq_length, c, h, w]
        batch_size, seq_length = x.size(0), x.size(1)        
        # Reshape to [batch_size * seq_length, c, h, w]
        x = x.reshape(batch_size * seq_length, x.size(2), x.size(3), x.size(4))

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)

        # Reshape to [batch_size, seq_length, hidden_size]
        x = x.reshape(batch_size, seq_length, -1)
        
        x, _ = self.lstm(x)  # LSTM input shape: [batch_size, seq_length, hidden_size]
        x = self.fc2(x)  # Output shape: [batch_size, seq_length, action_size]
        return x