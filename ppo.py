# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.distributions import Categorical


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def ppo_update(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    states,
    actions,
    log_probs_old,
    returns,
    advantages,
    epsilon=0.2,
    c1=0.5,
    c2=0.01,
):
    # Convert lists to tensors
    states = torch.stack(states)
    actions = torch.tensor(actions)
    log_probs_old = torch.tensor(log_probs_old)
    returns = torch.tensor(returns)
    advantages = torch.tensor(advantages)

    # Policy loss
    probs = policy_net(states)
    m = Categorical(probs)
    log_probs = m.log_prob(actions)
    ratio = torch.exp(log_probs - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    values = value_net(states).squeeze()
    value_loss = (returns - values).pow(2).mean()

    # Update policy network
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    # Update value network
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()


class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.is_terminals[:]

    def add(self, state, action, log_prob, reward, value, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.is_terminals.append(is_terminal)

    def get_batch(self):
        # Convert lists to PyTorch tensors
        states_tensor = torch.stack(self.states)
        actions_tensor = torch.stack(self.actions)
        log_probs_tensor = torch.stack(self.log_probs)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)
        values_tensor = torch.stack(self.values)
        is_terminals_tensor = torch.tensor(self.is_terminals, dtype=torch.float32)

        # Clear buffer
        self.clear()

        return (
            states_tensor,
            actions_tensor,
            log_probs_tensor,
            rewards_tensor,
            values_tensor,
            is_terminals_tensor,
        )
