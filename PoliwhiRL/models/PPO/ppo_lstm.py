import torch
import torch.nn as nn
from PoliwhiRL.models.CNN.GameBoy import GameBoyOptimizedCNN

class FlexibleInputLayer(nn.Module):
    def __init__(self, input_shape, d_model):
        super(FlexibleInputLayer, self).__init__()
        self.input_shape = input_shape
        self.d_model = d_model

        if len(input_shape) == 3:  # [C, H, W]
            self.cnn = GameBoyOptimizedCNN(input_shape, d_model)
        elif len(input_shape) == 2:  # [H, W]
            # Calculate total input size from height and width
            total_input_size = int(input_shape[0] * input_shape[1])
            self.fc = nn.Linear(total_input_size, d_model)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

    def forward(self, x):
        if len(self.input_shape) == 3:
            return self.cnn(x)
        else:
            # Ensure proper reshaping for 2D input
            batch_size = x.size(0)
            flattened = x.view(batch_size, -1)
            return torch.relu(self.fc(flattened))

class PPOLSTMPolicy(nn.Module):
    def __init__(self, input_shape, action_size, d_model=16, lstm_layers=1):
        super(PPOLSTMPolicy, self).__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.d_model = d_model
        
        # CNN or FC layer to process each frame
        self.flexible_input = FlexibleInputLayer(input_shape, d_model)
        
        # LSTM to process the sequence of encoded frames
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Output heads for actor and critic
        self.fc_actor = nn.Linear(d_model, action_size)
        self.fc_critic = nn.Linear(d_model, 1)
        
        # Initialize LSTM weights
        self._init_lstm_weights()

    def _init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        # Input shape: [batch_size, sequence_length, channels, height, width]
        batch_size, seq_len = x.size()[:2]
        
        # Reshape to process all frames through CNN: [batch_size * sequence_length, channels, height, width]
        x = x.view(batch_size * seq_len, *self.input_shape)
        
        # Process through CNN/FC
        x = self.flexible_input(x)
        
        # Reshape back to sequence: [batch_size, sequence_length, d_model]
        x = x.view(batch_size, seq_len, self.d_model)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use the final output of the sequence
        final_out = lstm_out[:, -1]
        
        # Actor head (action probabilities)
        action_probs = torch.softmax(self.fc_actor(final_out), dim=-1)
        
        # Critic head (state value)
        value = self.fc_critic(final_out)
        
        return action_probs, value

    def get_init_states(self, batch_size=1, device='cuda'):
        """Initialize LSTM hidden and cell states"""
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.d_model).to(device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.d_model).to(device)
        return (h0, c0)

    def forward_with_states(self, x, hidden_states=None):
        """Forward pass that also returns LSTM states for recurrent processing"""
        batch_size, seq_len = x.size()[:2]
        
        # Process through CNN/FC
        x = x.view(batch_size * seq_len, *self.input_shape)
        x = self.flexible_input(x)
        x = x.view(batch_size, seq_len, self.d_model)
        
        # Process through LSTM with states
        if hidden_states is None:
            hidden_states = self.get_init_states(batch_size, x.device)
        
        lstm_out, new_states = self.lstm(x, hidden_states)
        final_out = lstm_out[:, -1]
        
        # Actor and critic heads
        action_probs = torch.softmax(self.fc_actor(final_out), dim=-1)
        value = self.fc_critic(final_out)
        
        return action_probs, value, new_states