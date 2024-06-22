import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

class FeatureNN(nn.Module):
    def __init__(self, input_dim, vision=True, base_channels=32, num_conv_layers=3):
        super(FeatureNN, self).__init__()
        self.vision = vision
        
        if self.vision:
            layers = []
            in_channels = input_dim[0]
            for i in range(num_conv_layers):
                out_channels = base_channels * (2**i)
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1 if i == 0 else 2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(p=0.2)
                ])
                in_channels = out_channels
            
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # Global average pooling
            layers.append(nn.Flatten())
            
            self.feature_layer = nn.Sequential(*layers)
        else:
            self.feature_layer = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            )

    def feature_size(self, input_dim):
        if self.vision:
            return self.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1)
        else:
            return 1024  # The output size of the last linear layer

    def forward(self, x):
        return self.feature_layer(x)

class PPOModel(nn.Module):
    def __init__(self, input_dim, num_actions, vision=True, num_transformer_layers=3):
        super(PPOModel, self).__init__()
        self.vision = vision
        self.FeatureCNN = FeatureNN(input_dim, vision)
        feature_size = self.FeatureCNN.feature_size(input_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=feature_size, 
            nhead=8, 
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        self.actor = nn.Linear(feature_size, num_actions)
        self.critic = nn.Linear(feature_size, 1)

    def forward(self, x):
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
        
        # Apply transformer encoder
        features = self.transformer_encoder(features)
        
        # Use the last output of the Transformer
        features = features[:, -1, :]
        
        action_probs = F.softmax(self.actor(features), dim=-1)
        value_estimates = self.critic(features)
        
        return action_probs, value_estimates