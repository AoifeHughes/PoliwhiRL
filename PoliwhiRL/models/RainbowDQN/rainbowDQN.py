import torch
import torch.nn as nn
import torch.nn.functional as F
from .noisylinear import NoisyLinear  # Assuming you have this module for noisy layers

class RainbowDQN(nn.Module):
    def __init__(self, input_dim, num_actions, device, atom_size=51, Vmin=-10, Vmax=10):
        super(RainbowDQN, self).__init__()
        self.num_actions = num_actions
        self.atom_size = atom_size
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.device = device
        self.support = (
            torch.linspace(Vmin, Vmax, atom_size).view(1, 1, atom_size).to(device)
        )

        # Define network layers
        self.feature_layer = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4, padding=1),  # Adjusted for input dimensions
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_input_dim = self.feature_size(input_dim)

        # LSTM Layer
        self.lstm_hidden_size = 512  # Adjustable
        self.lstm = nn.LSTM(input_size=self.fc_input_dim, hidden_size=self.lstm_hidden_size, batch_first=True)
        
        # Initialize hidden and cell states
        self.hidden = None

        # Value and Advantage Stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 512), nn.ReLU(), nn.Linear(512, atom_size)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * atom_size),
        )

    def forward(self, x):
        x = self.feature_layer(x)
        x = x.view(x.size(0), 1, -1)  # Reshape x for LSTM processing
        
        # Pass through LSTM
        lstm_out, new_hidden = self.lstm(x, self.hidden)
        
        # Detach new hidden state from the graph
        new_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
        
        # Update self.hidden with the detached state for the next forward pass
        self.hidden = new_hidden
        
        x = lstm_out[:, -1, :]  # Only use the output for the last sequence element

        # Get Q-value distributions and compute q_values
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

    def init_hidden(self, batch_size):
        # Initializes hidden states of LSTM; call at the beginning of an episode
        self.hidden = (torch.zeros(1, batch_size, self.lstm_hidden_size).to(self.device),
                       torch.zeros(1, batch_size, self.lstm_hidden_size).to(self.device))

    def reset_noise(self):
        """Reset all noisy layers"""
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def reset_noise_and_hidden(self, batch_size):
        """Reset all noisy layers and hidden states"""
        self.reset_noise()
        self.init_hidden(batch_size)