# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from PoliwhiRL.models.transformers.positional_encoding import PositionalEncoding


class EnhancedGameBoyCNN(nn.Module):
    """Deeper CNN with residual connections and multi-scale features"""
    def __init__(self, input_shape, base_channels=32):
        super(EnhancedGameBoyCNN, self).__init__()
        in_channels = input_shape[0]
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Residual blocks at different scales
        self.layer1 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer2 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # Downsample if needed
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Multi-scale features
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        
        # Global pooling
        pooled = self.avgpool(feat3).flatten(1)
        
        # Also return intermediate features for skip connections
        return pooled, (feat1, feat2, feat3)


class ResidualBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpatialMemoryBank(nn.Module):
    """Dynamic spatial memory that can grow and includes coordinates"""
    def __init__(self, d_model, max_locations=1000):
        super(SpatialMemoryBank, self).__init__()
        self.d_model = d_model
        self.max_locations = max_locations
        
        # Learnable coordinate embeddings
        self.coord_embed = nn.Linear(3, d_model // 4)  # x, y, map_id
        self.visit_embed = nn.Embedding(100, d_model // 4)  # Visit count embedding
        self.time_embed = nn.Linear(1, d_model // 4)  # Time since last visit
        
        # Combine all spatial features
        self.spatial_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Attention for memory retrieval
        self.memory_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        
    def forward(self, memory_tensor, current_coords=None):
        # memory_tensor: [batch, locations, features]
        # current_coords: [batch, 3] (x, y, map_id)
        
        if memory_tensor.size(1) == 0:
            return torch.zeros(memory_tensor.size(0), self.d_model).to(memory_tensor.device)
            
        # Split memory tensor into components
        coords = memory_tensor[:, :, :3]  # x, y, map
        visits = memory_tensor[:, :, 3].long().clamp(0, 99)  # Visit count
        time_since = memory_tensor[:, :, 4:5]  # Time component
        
        # Embed each component
        coord_emb = self.coord_embed(coords)
        visit_emb = self.visit_embed(visits)
        time_emb = self.time_embed(time_since)
        
        # Concatenate and process
        spatial_features = torch.cat([coord_emb, visit_emb, time_emb, 
                                     memory_tensor[:, :, 5:]], dim=-1)
        spatial_features = self.spatial_mlp(spatial_features)
        
        # If current coordinates provided, use as query
        if current_coords is not None:
            query = self.coord_embed(current_coords).unsqueeze(1)
            attended, _ = self.memory_attention(query, spatial_features, spatial_features)
            return attended.squeeze(1)
        else:
            # Global pooling
            return spatial_features.mean(dim=1)


class GameStateEncoder(nn.Module):
    """Encode game-specific state information"""
    def __init__(self, d_model):
        super(GameStateEncoder, self).__init__()
        # Different encoders for different game contexts
        self.menu_encoder = nn.Linear(64, d_model)  # Menu state features
        self.battle_encoder = nn.Linear(128, d_model)  # Battle state features
        self.inventory_encoder = nn.Linear(32, d_model)  # Inventory features
        
        # Context classifier (menu, overworld, battle, etc.)
        self.context_classifier = nn.Linear(d_model, 4)
        
    def forward(self, visual_features, ram_features=None):
        # This would need RAM reading to work properly
        # For now, return visual features
        return visual_features


class PPOTransformerEnhanced(nn.Module):
    """Enhanced transformer for complex Pokemon gameplay"""
    def __init__(self, input_shape, action_size, d_model=256, nhead=16, 
                 num_layers=6, sequence_length=32, **kwargs):
        super(PPOTransformerEnhanced, self).__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Enhanced visual encoder
        base_channels = kwargs.get('base_channels', 32)
        self.visual_encoder = EnhancedGameBoyCNN(input_shape, base_channels)
        
        # Calculate CNN output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            cnn_out, _ = self.visual_encoder(dummy_input)
            cnn_dim = cnn_out.shape[1]
            
        # Project CNN features to model dimension
        self.visual_projection = nn.Linear(cnn_dim, d_model)
        
        # Positional encoding for sequences
        self.pos_encoder = PositionalEncoding(d_model, max_len=2000)
        
        # Spatial memory bank
        self.spatial_memory = SpatialMemoryBank(d_model)
        
        # Game state encoder
        self.game_state_encoder = GameStateEncoder(d_model)
        
        # Transformer with more capacity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Separate transformers for actor and critic (better stability)
        self.actor_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead // 2, d_model * 2, 0.1, batch_first=True),
            num_layers=2
        )
        self.critic_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead // 2, d_model * 2, 0.1, batch_first=True),
            num_layers=2
        )
        
        # Output heads
        self.actor_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_size)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Auxiliary outputs for better learning
        self.next_frame_predictor = nn.Linear(d_model, cnn_dim)  # Predict next visual features
        self.reward_predictor = nn.Linear(d_model, 1)  # Predict immediate reward
        
    def forward(self, x, exploration_tensor=None, return_auxiliaries=False):
        batch_size, seq_len = x.size()[:2]
        
        # Process visual features for each timestep
        visual_features = []
        for t in range(seq_len):
            frame = x[:, t].view(batch_size, *self.input_shape)
            pooled, multi_scale = self.visual_encoder(frame)
            visual_features.append(pooled)
            
        visual_features = torch.stack(visual_features, dim=1)  # [batch, seq, cnn_dim]
        visual_features = self.visual_projection(visual_features)  # [batch, seq, d_model]
        
        # Add positional encoding
        encoded_sequence = self.pos_encoder(visual_features)
        
        # Process with main transformer
        transformer_out = self.transformer_encoder(encoded_sequence)
        
        # Get final representation
        final_features = transformer_out[:, -1, :]  # [batch, d_model]
        
        # Process exploration memory if available
        if exploration_tensor is not None and exploration_tensor.size(1) > 0:
            # Assume exploration_tensor contains [x, y, map, visits, time, ...]
            memory_features = self.spatial_memory(exploration_tensor)
        else:
            memory_features = torch.zeros_like(final_features)
            
        # Combine features
        combined = torch.cat([final_features, memory_features], dim=-1)
        
        # Separate processing for actor and critic
        actor_input = self.actor_transformer(transformer_out)[:, -1, :]
        critic_input = self.critic_transformer(transformer_out)[:, -1, :]
        
        actor_combined = torch.cat([actor_input, memory_features], dim=-1)
        critic_combined = torch.cat([critic_input, memory_features], dim=-1)
        
        # Get outputs
        action_logits = self.actor_head(actor_combined)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic_head(critic_combined)
        
        if return_auxiliaries:
            # Auxiliary predictions for better representation learning
            next_frame_pred = self.next_frame_predictor(final_features)
            reward_pred = self.reward_predictor(final_features)
            return action_probs, value, next_frame_pred, reward_pred
            
        return action_probs, value
        
    def reset_actor(self):
        """Reset actor components while preserving critic and representations"""
        # Reset actor transformer
        for layer in self.actor_transformer.layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
                    
        # Reset actor head
        for module in self.actor_head:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        print("Enhanced actor components reset")


# Configuration suggestions for enhanced model
ENHANCED_CONFIG = {
    "d_model": 256,  # Larger model dimension
    "nhead": 16,     # More attention heads
    "num_layers": 6, # Deeper transformer
    "sequence_length": 32,  # Longer temporal context
    "base_channels": 32,    # CNN base channels
    "use_auxiliary_losses": True,  # Enable auxiliary losses
    "auxiliary_loss_weight": 0.1,  # Weight for auxiliary losses
}