import torch.nn as nn
import torch
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dim=256):
        super(ActorCritic, self).__init__()

        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out_size(input_dims)

        self.lstm = nn.LSTM(conv_out_size, hidden_dim, batch_first=True)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def _get_conv_out_size(self, input_dims):
        return self.conv(torch.zeros(1, 1, *input_dims)).shape[1]

    def forward(self, state, hidden):
        conv_out = self.conv(state.unsqueeze(1))  # Add channel dimension
        lstm_out, hidden = self.lstm(conv_out.unsqueeze(1), hidden)
        action_probs = self.actor(lstm_out.squeeze(1))
        value = self.critic(lstm_out.squeeze(1))
        return action_probs, value, hidden

    def act(self, state, hidden):
        action_probs, value, hidden = self.forward(state, hidden)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.hidden_dim),
            torch.zeros(1, batch_size, self.hidden_dim),
        )