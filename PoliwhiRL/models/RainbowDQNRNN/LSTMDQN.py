import torch
import torch.nn as nn
import torch.nn.functional as F
from .noisylinear import NoisyLinear

class LSTMDQN(nn.Module):
    def __init__(self, input_dim, num_actions, device, atom_size=51, Vmin=-10, Vmax=10, sequence_length=4):
        super(LSTMDQN, self).__init__()
        self.num_actions = num_actions
        self.atom_size = atom_size
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.sequence_length = sequence_length
        self.support = torch.linspace(Vmin, Vmax, atom_size).view(1, 1, atom_size).to(device)
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
        self.lstm = nn.LSTM(self.fc_input_dim, 512, batch_first=True)
        
        # Assuming you want to directly output Q-values for each action at each timestep
        self.value_head = nn.Linear(512, 1)  # Outputs a single value for V(s)
        self.advantage_head = nn.Linear(512, num_actions)  # Outputs advantage for each action

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.feature_layer(x)
        x = x.view(batch_size, sequence_length, -1)
        lstm_out, _ = self.lstm(x)
        
        # Process each timestep output through the value and advantage streams
        values = self.value_head(lstm_out)  # Shape: [batch_size, sequence_length, 1]
        advantages = self.advantage_head(lstm_out)  # Shape: [batch_size, sequence_length, num_actions]

        # Use the Dueling Network architecture to calculate Q-values from values and advantages
        q_values = values + (advantages - advantages.mean(dim=2, keepdim=True))

        return q_values

    def feature_size(self, input_dim):
        return self.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1)

    def reset_noise(self):
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
