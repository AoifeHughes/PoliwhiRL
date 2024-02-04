import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 
class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration."""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class RainbowDQN(nn.Module):
    def __init__(self, input_dim, num_actions, device, atom_size=51, Vmin=-10, Vmax=10):
        super(RainbowDQN, self).__init__()
        self.num_actions = num_actions
        self.atom_size = atom_size
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.support = torch.linspace(Vmin, Vmax, atom_size).view(1, 1, atom_size).to(device)

        # Define network layers
        self.feature_layer = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4, padding=1), # Adjusted for input dimensions
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_input_dim = self.feature_size(input_dim)

        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, atom_size)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * atom_size)
        )

    def forward(self, x):
        dist = self.get_distribution(x)
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values

    def get_distribution(self, x):
        x = self.feature_layer(x)
        value = self.value_stream(x).view(-1, 1, self.atom_size)
        advantage = self.advantage_stream(x).view(-1, self.num_actions, self.atom_size)
        advantage_mean = advantage.mean(1, keepdim=True)
        dist = value + advantage - advantage_mean
        dist = F.softmax(dist, dim=-1)
        dist = dist.clamp(min=1e-3)  # Avoid zeros
        return dist

    def feature_size(self, input_dim):
        return self.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1)


    def reset_noise(self):
        """Reset all noisy layers"""
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves the current state of training."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename="checkpoint.pth.tar", device="cpu"):
    """Loads the checkpoint and returns the state."""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        print(f"Checkpoint loaded from {filename}")
        return checkpoint
    else:
        print(f"Checkpoint file not found: {filename}")
        return None