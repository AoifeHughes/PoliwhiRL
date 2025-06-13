# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F


class GameBoyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GameBoyBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class GameBoyOptimizedCNN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(GameBoyOptimizedCNN, self).__init__()
        # Enhanced CNN for better visual understanding
        self.block1 = GameBoyBlock(input_shape[0], 32)  # 16→32 channels
        self.block2 = GameBoyBlock(32, 64)              # 32→64 channels  
        self.block3 = GameBoyBlock(64, 128)             # Add 3rd block for richer features
        
        # Add residual connection for deeper learning
        self.residual_conv = nn.Conv2d(input_shape[0], 128, kernel_size=1, stride=8)
        
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            sample_output = self.block3(self.block2(self.block1(sample_input)))
            self.flat_features = sample_output.view(1, -1).size(1)
        
        # Multi-layer feature extraction 
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, x):
        # Main pathway
        identity = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Residual connection
        identity = self.residual_conv(identity)
        if identity.shape != x.shape:
            # Adaptive pooling if shapes don't match
            identity = F.adaptive_avg_pool2d(identity, x.shape[2:])
        
        x = x + identity
        x = x.view(-1, self.flat_features)
        return self.fc_layers(x)
