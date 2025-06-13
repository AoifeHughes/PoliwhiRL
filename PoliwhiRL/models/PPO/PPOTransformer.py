# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from PoliwhiRL.models.CNN.GameBoy import GameBoyOptimizedCNN
from PoliwhiRL.models.transformers.positional_encoding import PositionalEncoding


class FlexibleInputLayer(nn.Module):
    def __init__(self, input_shape, d_model):
        super(FlexibleInputLayer, self).__init__()
        self.input_shape = input_shape
        self.d_model = d_model

        if len(input_shape) == 3:  # [C, H, W]
            self.cnn = GameBoyOptimizedCNN(input_shape, d_model)
        elif len(input_shape) == 2:  # [X, Y]
            self.fc = nn.Linear(input_shape[0] * input_shape[1], d_model)
        else:
            raise ValueError("Unsupported input shape")

    def forward(self, x):
        if len(self.input_shape) == 3:
            return self.cnn(x)
        else:
            return torch.relu(self.fc(x.view(x.size(0), -1)))


class ExplorationEncoder(nn.Module):
    def __init__(self, d_model, history_length=5):
        super(ExplorationEncoder, self).__init__()
        # Enhanced exploration understanding
        input_features = 1 + history_length
        self.fc1 = nn.Linear(input_features, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        
        # Multi-head attention for better spatial understanding
        self.attention = nn.MultiheadAttention(128, 8, batch_first=True)
        self.layer_norm = nn.LayerNorm(128)
        
        # Output projection to match d_model
        self.fc_out = nn.Linear(128, d_model)

    def forward(self, x):
        # x shape: [batch_size, num_locations, 1+history_length]
        batch_size = x.size(0)
        
        # Enhanced feature extraction
        x = torch.relu(self.fc1(x))     # [batch_size, num_locations, 32]
        x = torch.relu(self.fc2(x))     # [batch_size, num_locations, 64]  
        x = torch.relu(self.fc3(x))     # [batch_size, num_locations, 128]

        # Self-attention over locations with normalization
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(attn_output + x)  # Residual connection

        # Global pooling over locations
        x = x.mean(dim=1)  # [batch_size, 128]

        # Project to d_model
        x = self.fc_out(x)  # [batch_size, d_model]
        return x


class PPOTransformer(nn.Module):
    def __init__(
        self, input_shape, action_size, d_model=256, nhead=8, num_layers=6, **kwargs
    ):
        super(PPOTransformer, self).__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.d_model = d_model

        self.flexible_input = FlexibleInputLayer(input_shape, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        # Get history length from config or use default
        history_length = kwargs.get("ppo_exploration_history_length", 5)
        # Exploration memory encoder
        self.exploration_encoder = ExplorationEncoder(
            d_model, history_length=history_length
        )
        
        # Goal conditioning system for curriculum learning
        self.goal_embedding = nn.Embedding(8, d_model // 4)  # Support up to 8 goals
        self.goal_projection = nn.Linear(d_model // 4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Enhanced Actor head with deeper layers for complex decision making
        self.actor_layers = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_size)
        )
        
        # Enhanced Critic head with separate value estimation pathway
        self.critic_layers = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, exploration_tensor=None, goal_stage=None):
        batch_size, seq_len = x.size()[:2]
        x = x.view(batch_size * seq_len, *self.input_shape)
        x = self.flexible_input(x)
        x = x.view(batch_size, seq_len, self.d_model)
        
        # Add goal conditioning if provided
        if goal_stage is not None:
            goal_embed = self.goal_embedding(goal_stage)
            goal_features = self.goal_projection(goal_embed)
            # Add goal information to each timestep
            goal_features = goal_features.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + goal_features
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Use the last output of the sequence
        x = x[:, -1, :]

        # Process exploration memory if provided
        if exploration_tensor is not None:
            # Ensure exploration_tensor is the right shape [batch_size, num_locations, 1+history_length]
            if (
                len(exploration_tensor.shape) == 3
                and exploration_tensor.shape[0] == batch_size
            ):
                exploration_features = self.exploration_encoder(exploration_tensor)
                # Concatenate with transformer output
                combined_features = torch.cat([x, exploration_features], dim=1)
            else:
                # If exploration tensor is not properly shaped, just duplicate the transformer output
                combined_features = torch.cat([x, x], dim=1)
        else:
            # If no exploration tensor, just duplicate the transformer output
            combined_features = torch.cat([x, x], dim=1)
        
        # Add goal conditioning to final features if available
        if goal_stage is not None:
            goal_features = self.goal_projection(self.goal_embedding(goal_stage))
            combined_features = torch.cat([combined_features, goal_features], dim=1)
            # Add linear layer to handle increased dimensionality
            if not hasattr(self, 'goal_adaptation_layer'):
                self.goal_adaptation_layer = nn.Linear(
                    combined_features.size(-1), self.d_model * 2
                ).to(combined_features.device)
            combined_features = self.goal_adaptation_layer(combined_features)

        action_logits = self.actor_layers(combined_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        value = self.critic_layers(combined_features)

        return action_probs, value
    
    def reset_actor(self):
        """Reset actor layers to random initialization while preserving critic"""
        for layer in self.actor_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif hasattr(layer, 'weight'):
                # For layers without reset_parameters, reinitialize manually
                nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        print("Actor layers reset to random initialization")
