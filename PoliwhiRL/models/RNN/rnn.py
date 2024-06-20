import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Todo fix this properly later
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, num_actions):
        super(RNN, self).__init__()
        
        # CNN layers shared by both policy and value networks
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        
        # Flattening layer
        self._to_linear = None
        self.convs(torch.randn(1, 3, 160, 144))
        
        # LSTM layer shared by both policy and value networks
        self.lstm = nn.LSTM(input_size=self._to_linear, hidden_size=512, num_layers=1, batch_first=True)
        
        # Policy network layers
        self.policy_fc = nn.Linear(in_features=512, out_features=num_actions)
        
        # Value network layers
        self.value_fc = nn.Linear(in_features=512, out_features=1)
    
    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if self._to_linear is None:
            self._to_linear = int(np.prod(x.size()[1:]))
        return x.view(x.size(0), -1)
    
    def forward(self, x, hidden):
        if x.dim() == 4:
            x = x.unsqueeze(0) # add batch size if needed
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Apply CNN
        x = self.convs(x)
        
        # Prepare for LSTM
        x = x.view(batch_size, seq_len, -1)
        
        # Apply LSTM
        x, hidden = self.lstm(x, hidden)
        
        # Take the last output from the sequence
        x = x[:, -1, :]
        
        # Policy network output
        policy_logits = self.policy_fc(x)
        
        # Value network output
        state_value = self.value_fc(x)
        
        return policy_logits, state_value, hidden
    
    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state for LSTM
        return (torch.zeros(1, batch_size, 512).to(device), torch.zeros(1, batch_size, 512).to(device))
