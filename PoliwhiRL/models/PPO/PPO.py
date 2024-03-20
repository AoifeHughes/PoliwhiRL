class Actor(nn.Module):
    def __init__(self, feature_dim, num_actions):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=512, batch_first=True)
        self.policy_head = nn.Linear(512, num_actions)
        
    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        return F.softmax(self.policy_head(lstm_out[:, -1, :]), dim=-1)

class Critic(nn.Module):
    def __init__(self, feature_dim):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=512, batch_first=True)
        self.value_head = nn.Linear(512, 1)
        
    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        return self.value_head(lstm_out[:, -1, :])


class PPOModel(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(PPOModel, self).__init__()
        self.shared_cnn = SharedCNN(input_dim)
        self.actor = Actor(self.shared_cnn.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1), num_actions)
        self.critic = Critic(self.shared_cnn.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1))
        
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height, width)
        shared_features = self.shared_cnn(x)
        shared_features = shared_features.view(batch_size, seq_len, -1)
        action_probs = self.actor(shared_features)
        value_estimates = self.critic(shared_features)
        return action_probs, value_estimates
