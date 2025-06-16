# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


class GameStateEncoder(nn.Module):
    """
    Encodes Pokemon game state variables into meaningful features for the PPO model.
    
    Processes RAM variables including party info, progress metrics, location data,
    and collision states into a compact feature representation.
    """
    
    def __init__(self, feature_dim=64):
        super(GameStateEncoder, self).__init__()
        self.feature_dim = feature_dim
        
        # Define input feature dimensions based on RAM variables
        # Party info: level_sum, hp_sum, exp_sum, party_size (4 features)
        # Progress: pokedex_seen, pokedex_owned, money (3 features) 
        # Location: x, y, map_num, room, warp_number, map_bank (6 features)
        # Collision: up, down, left, right (4 features)
        self.input_dim = 17  # Total features from game state
        
        # Normalization layers for different feature groups
        self.party_norm = nn.LayerNorm(4)
        self.progress_norm = nn.LayerNorm(3)
        self.location_norm = nn.LayerNorm(6)
        self.collision_norm = nn.LayerNorm(4)
        
        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, self.feature_dim),
            nn.ReLU()
        )
        
        # Separate embedding for categorical features
        self.map_embedding = nn.Embedding(256, 16)  # Support up to 256 maps
        self.room_embedding = nn.Embedding(256, 8)   # Support up to 256 rooms
        
        # Final projection layer
        self.output_projection = nn.Linear(self.feature_dim + 24, self.feature_dim)
        
    def forward(self, game_state_dict):
        """
        Forward pass through game state encoder.
        
        Args:
            game_state_dict: Dictionary containing game state variables from RAM
            
        Returns:
            torch.Tensor: Encoded game state features [batch_size, feature_dim]
        """
        batch_size = len(game_state_dict) if isinstance(game_state_dict, list) else 1
        
        # Handle both single state and batch of states
        if not isinstance(game_state_dict, list):
            game_state_dict = [game_state_dict]
        
        # Extract and normalize features for each state in batch
        batch_features = []
        batch_map_embeds = []
        batch_room_embeds = []
        
        # Get the device from the model parameters
        device = next(self.parameters()).device
        
        for state in game_state_dict:
            # Extract party information
            party_info = state.get("party_info", (0, 0, 0))
            party_features = torch.tensor([
                party_info[0],  # total_level
                party_info[1],  # total_hp
                party_info[2],  # total_exp
                len(party_info) if isinstance(party_info, (list, tuple)) else 1  # party_size proxy
            ], dtype=torch.float32, device=device)
            
            # Extract progress information
            progress_features = torch.tensor([
                state.get("pokedex_seen", 0),
                state.get("pokedex_owned", 0),
                min(state.get("money", 0), 999999)  # Cap money to reasonable range
            ], dtype=torch.float32, device=device)
            
            # Extract location information
            location_features = torch.tensor([
                state.get("X", 0),
                state.get("Y", 0),
                state.get("map_num", 0),
                state.get("room", 0),
                state.get("warp_number", 0),
                state.get("map_bank", 0)
            ], dtype=torch.float32, device=device)
            
            # Extract collision information
            collision_features = torch.tensor([
                state.get("collision_up", 0),
                state.get("collision_down", 0), 
                state.get("collision_left", 0),
                state.get("collision_right", 0)
            ], dtype=torch.float32, device=device)
            
            # Normalize feature groups
            party_norm = self.party_norm(party_features.unsqueeze(0)).squeeze(0)
            progress_norm = self.progress_norm(progress_features.unsqueeze(0)).squeeze(0)
            location_norm = self.location_norm(location_features.unsqueeze(0)).squeeze(0)
            collision_norm = self.collision_norm(collision_features.unsqueeze(0)).squeeze(0)
            
            # Concatenate normalized features
            combined_features = torch.cat([
                party_norm, progress_norm, location_norm, collision_norm
            ])
            
            batch_features.append(combined_features)
            
            # Get embeddings for categorical features
            map_id = min(int(state.get("map_num", 0)), 255)
            room_id = min(int(state.get("room", 0)), 255)
            
            batch_map_embeds.append(self.map_embedding(torch.tensor(map_id, device=device)))
            batch_room_embeds.append(self.room_embedding(torch.tensor(room_id, device=device)))
        
        # Stack batch
        if batch_size == 1:
            features = batch_features[0].unsqueeze(0)
            map_embeds = batch_map_embeds[0].unsqueeze(0)
            room_embeds = batch_room_embeds[0].unsqueeze(0)
        else:
            features = torch.stack(batch_features)
            map_embeds = torch.stack(batch_map_embeds)
            room_embeds = torch.stack(batch_room_embeds)
        
        # Features are already on the correct device
        
        # Extract features through main network
        encoded_features = self.feature_net(features)
        
        # Combine with categorical embeddings
        combined_with_embeds = torch.cat([encoded_features, map_embeds, room_embeds], dim=-1)
        
        # Final projection
        output_features = self.output_projection(combined_with_embeds)
        
        return output_features
    
    def get_feature_importance(self):
        """
        Returns the relative importance of different feature groups.
        Useful for debugging and interpretability.
        """
        return {
            "party_info": "Pokemon levels, HP, experience - combat readiness",
            "progress": "Pokedex completion, money - game progression", 
            "location": "Current position and map context - spatial awareness",
            "collision": "Movement constraints - navigation planning",
            "embeddings": "Learned map/room representations - contextual understanding"
        }