# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from PoliwhiRL.models.CNN.GameBoy import GameBoyOptimizedCNN
from PoliwhiRL.models.transformers.positional_encoding import PositionalEncoding
from .game_state_encoder import GameStateEncoder
from .macro_action_module import MacroActionModule


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
        self.d_model = d_model
        
        # Enhanced input features for hierarchical memory
        # Support both old format (1+history_length) and new format (8 features)
        # Auto-detect based on input tensor shape
        self.supports_hierarchical = True
        self.old_input_features = 1 + history_length  # Original format
        self.new_input_features = 8  # Hierarchical format
        
        # Feature embedding layers for old format
        self.old_feature_embedding = nn.Sequential(
            nn.Linear(self.old_input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Feature embedding layers for new hierarchical format
        self.new_feature_embedding = nn.Sequential(
            nn.Linear(self.new_input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Positional encoding for location sequences
        self.positional_encoding = nn.Parameter(torch.randn(1000, 128) * 0.1)
        
        # Multi-head attention with importance-weighted attention
        self.importance_attention = nn.MultiheadAttention(128, 8, batch_first=True)
        self.spatial_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(128)
        
        # Importance-based weighting
        self.importance_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Memory compression for variable-length sequences
        self.memory_compressor = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, d_model)
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: [batch_size, num_locations, features] - auto-detect format
        batch_size, num_locations, num_features = x.size()
        
        # Auto-detect input format and use appropriate embedding
        if num_features == self.new_input_features:
            # New hierarchical format
            embedded = self.new_feature_embedding(x)  # [batch_size, num_locations, 128]
            use_advanced_features = True
        elif num_features == self.old_input_features:
            # Old format - fallback to simpler processing
            embedded = self.old_feature_embedding(x)  # [batch_size, num_locations, 128]
            use_advanced_features = False
        else:
            raise ValueError(f"Unexpected input feature size: {num_features}. Expected {self.old_input_features} or {self.new_input_features}")
        
        # Apply enhanced processing only for hierarchical format
        if not use_advanced_features:
            # Simple processing for old format
            # Just use mean pooling like the original
            pooled_features = embedded.mean(dim=1)  # [batch_size, 128]
            output = self.memory_compressor(pooled_features)  # [batch_size, d_model]
            return self.output_projection(output)
            
        # Advanced processing for hierarchical format continues below...
        
        # Add positional encoding based on sequence order
        if num_locations <= self.positional_encoding.size(0):
            pos_enc = self.positional_encoding[:num_locations].unsqueeze(0).expand(batch_size, -1, -1)
            embedded = embedded + pos_enc
        
        # Importance-weighted attention
        importance_weighted, importance_weights = self.importance_attention(embedded, embedded, embedded)
        embedded = self.layer_norm1(importance_weighted + embedded)
        
        # Spatial relationship attention
        spatial_attended, spatial_weights = self.spatial_attention(embedded, embedded, embedded)
        embedded = self.layer_norm2(spatial_attended + embedded)
        
        # Compute importance gates for each location
        importance_gates = self.importance_gate(embedded)  # [batch_size, num_locations, 1]
        
        # Apply importance weighting
        weighted_features = embedded * importance_gates
        
        # Adaptive pooling based on importance
        importance_sum = importance_gates.sum(dim=1, keepdim=True) + 1e-8
        pooled_features = weighted_features.sum(dim=1) / importance_sum.squeeze(-1)  # [batch_size, 128]
        
        # Memory compression
        compressed = self.memory_compressor(pooled_features)  # [batch_size, d_model]
        
        # Final output projection
        output = self.output_projection(compressed)
        
        return output
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability."""
        with torch.no_grad():
            batch_size, num_locations, num_features = x.size()
            
            # Only return attention weights for hierarchical format
            if num_features != self.new_input_features:
                return {'message': 'Attention weights only available for hierarchical memory format'}
            
            embedded = self.new_feature_embedding(x)
            
            if num_locations <= self.positional_encoding.size(0):
                pos_enc = self.positional_encoding[:num_locations].unsqueeze(0).expand(batch_size, -1, -1)
                embedded = embedded + pos_enc
            
            _, importance_weights = self.importance_attention(embedded, embedded, embedded)
            _, spatial_weights = self.spatial_attention(embedded, embedded, embedded)
            importance_gates = self.importance_gate(embedded)
            
            return {
                'importance_attention': importance_weights,
                'spatial_attention': spatial_weights, 
                'importance_gates': importance_gates
            }


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
        
        # Game state encoder for RAM variables
        game_state_feature_dim = kwargs.get("game_state_feature_dim", 64)
        self.use_game_state = kwargs.get("use_game_state_features", True)
        if self.use_game_state:
            self.game_state_encoder = GameStateEncoder(feature_dim=game_state_feature_dim)
        
        # Macro action learning system
        self.use_macro_actions = kwargs.get("use_macro_actions", False)
        max_macro_length = kwargs.get("max_macro_action_length", 5)
        if self.use_macro_actions:
            self.macro_action_module = MacroActionModule(
                action_space_size=action_size,
                max_sequence_length=max_macro_length,
                d_model=d_model,
                device=kwargs.get("device", "cpu")
            )
            # Extend action space to include potential macro actions
            self.extended_action_size = action_size + 1000  # Reserve space for macro actions
        else:
            self.extended_action_size = action_size
        
        # Goal conditioning system for curriculum learning
        self.goal_embedding = nn.Embedding(8, d_model // 4)  # Support up to 8 goals
        self.goal_projection = nn.Linear(d_model // 4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Calculate input dimension for actor/critic heads
        # Base: d_model * 2 (visual + exploration)
        # + game_state_feature_dim (if using game state)
        actor_critic_input_dim = d_model * 2
        if self.use_game_state:
            actor_critic_input_dim += game_state_feature_dim
        
        # Enhanced Actor head with deeper layers for complex decision making
        # Support both primitive and macro actions
        self.primitive_actor = nn.Sequential(
            nn.Linear(actor_critic_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_size)
        )
        
        # Macro action head (only if macro actions are enabled)
        if self.use_macro_actions:
            self.macro_actor = nn.Sequential(
                nn.Linear(actor_critic_input_dim, d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1000)  # Max macro actions
            )
            
            # Action type selector (primitive vs macro)
            self.action_type_selector = nn.Sequential(
                nn.Linear(actor_critic_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),  # [primitive_prob, macro_prob]
                nn.Softmax(dim=-1)
            )
        
        # Enhanced Critic head with separate value estimation pathway
        self.critic_layers = nn.Sequential(
            nn.Linear(actor_critic_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, exploration_tensor=None, goal_stage=None, game_state=None):
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
        
        # Process game state if provided and enabled
        if self.use_game_state and game_state is not None:
            game_state_features = self.game_state_encoder(game_state)
            # Concatenate game state features
            combined_features = torch.cat([combined_features, game_state_features], dim=1)
        elif self.use_game_state:
            # If game state is enabled but not provided, create zero features
            device = combined_features.device
            game_state_feature_dim = self.game_state_encoder.feature_dim
            zero_game_state = torch.zeros(batch_size, game_state_feature_dim, device=device)
            combined_features = torch.cat([combined_features, zero_game_state], dim=1)
        
        # Add goal conditioning to final features if available
        if goal_stage is not None:
            goal_features = self.goal_projection(self.goal_embedding(goal_stage))
            combined_features = torch.cat([combined_features, goal_features], dim=1)
            # Add linear layer to handle increased dimensionality
            expected_dim = self.d_model * 2
            if self.use_game_state:
                expected_dim += self.game_state_encoder.feature_dim
            if not hasattr(self, 'goal_adaptation_layer'):
                self.goal_adaptation_layer = nn.Linear(
                    combined_features.size(-1), expected_dim
                ).to(combined_features.device)
            combined_features = self.goal_adaptation_layer(combined_features)

        # Compute value estimate
        value = self.critic_layers(combined_features)
        
        # Hierarchical action selection
        if self.use_macro_actions:
            # Get action type probabilities
            action_type_probs = self.action_type_selector(combined_features)  # [batch, 2]
            
            # Get primitive action probabilities
            primitive_logits = self.primitive_actor(combined_features)
            primitive_probs = torch.softmax(primitive_logits, dim=-1)
            
            # Get macro action probabilities (only for active macros)
            macro_logits = self.macro_actor(combined_features)
            
            # Mask macro actions based on what's available in the macro module
            if hasattr(self.macro_action_module, 'macro_sequences'):
                available_macros = list(self.macro_action_module.macro_sequences.keys())
                macro_mask = torch.zeros_like(macro_logits)
                for macro_id in available_macros:
                    if macro_id - self.action_size < macro_mask.size(-1):
                        macro_mask[:, macro_id - self.action_size] = 1.0
                macro_logits = macro_logits * macro_mask + (macro_mask - 1) * 1e9  # Mask unavailable
            
            macro_probs = torch.softmax(macro_logits, dim=-1)
            
            # Combine probabilities
            primitive_weight = action_type_probs[:, 0:1]  # [batch, 1]
            macro_weight = action_type_probs[:, 1:2]  # [batch, 1]
            
            # Create combined action probability distribution
            batch_size = combined_features.size(0)
            total_actions = self.extended_action_size
            action_probs = torch.zeros(batch_size, total_actions, device=combined_features.device)
            
            # Fill primitive actions
            action_probs[:, :self.action_size] = primitive_probs * primitive_weight
            
            # Fill macro actions
            if hasattr(self.macro_action_module, 'macro_sequences') and self.macro_action_module.macro_sequences:
                available_macros = list(self.macro_action_module.macro_sequences.keys())
                for i, macro_id in enumerate(available_macros):
                    if macro_id < total_actions:
                        action_probs[:, macro_id] = macro_probs[:, i] * macro_weight.squeeze()
                        
        else:
            # Standard primitive action selection
            action_logits = self.primitive_actor(combined_features)
            action_probs = torch.softmax(action_logits, dim=-1)

        return action_probs, value
    
    def reset_actor(self):
        """Reset actor layers to random initialization while preserving critic"""
        # Reset primitive actor
        for layer in self.primitive_actor:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif hasattr(layer, 'weight'):
                # For layers without reset_parameters, reinitialize manually
                nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Reset macro actor if enabled
        if self.use_macro_actions:
            for layer in self.macro_actor:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                elif hasattr(layer, 'weight'):
                    nn.init.xavier_uniform_(layer.weight)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                        
            for layer in self.action_type_selector:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                elif hasattr(layer, 'weight'):
                    nn.init.xavier_uniform_(layer.weight)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        print("Actor layers reset to random initialization")
