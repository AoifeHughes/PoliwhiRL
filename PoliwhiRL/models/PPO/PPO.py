# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, h, w, outputs, USE_GRAYSCALE):
        super(PolicyNetwork, self).__init__()
        self.USE_GRAYSCALE = USE_GRAYSCALE
        self.conv1 = nn.Conv2d(1 if USE_GRAYSCALE else 3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(64)

        self._to_linear = None
        self._compute_conv_output_size(h, w)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, outputs)

    def _compute_conv_output_size(self, h, w):
        x = torch.rand(1, 1 if self.USE_GRAYSCALE else 3, h, w)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, h, w, USE_GRAYSCALE):
        super(ValueNetwork, self).__init__()
        self.USE_GRAYSCALE = USE_GRAYSCALE
        self.conv1 = nn.Conv2d(1 if USE_GRAYSCALE else 3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(64)

        self._to_linear = None
        self._compute_conv_output_size(h, w)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 1)  # Outputs a single value for the value function

    def _compute_conv_output_size(self, h, w):
        x = torch.rand(1, 1 if self.USE_GRAYSCALE else 3, h, w)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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
        self.returns = []
        self.advantages = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.is_terminals[:]
        del self.returns[:]
        del self.advantages[:]

    def add(self, state, action, log_prob, reward, value, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.is_terminals.append(is_terminal)

    def compute_gae_and_returns(self, next_value, gamma=0.99, tau=0.95):
        gae = 0
        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + gamma * next_value * (1 - self.is_terminals[step])
                - self.values[step]
            )
            gae = delta + gamma * tau * (1 - self.is_terminals[step]) * gae
            self.returns.insert(0, gae + self.values[step])
            next_value = self.values[step]
        self.returns = torch.tensor(self.returns, dtype=torch.float32).detach()
        self.advantages = (
            self.returns - torch.tensor(self.values, dtype=torch.float32).detach()
        )

    def get_batch(self):
        # Ensure to call compute_gae_and_returns before this to populate returns and advantages
        states_tensor = torch.stack(self.states)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long)
        log_probs_tensor = torch.stack(self.log_probs)
        returns_tensor = self.returns
        advantages_tensor = self.advantages
        is_terminals_tensor = torch.tensor(self.is_terminals, dtype=torch.float32)

        self.clear()  # Clear buffer after getting the batch

        return (
            states_tensor,
            actions_tensor,
            log_probs_tensor,
            returns_tensor,
            advantages_tensor,
            is_terminals_tensor,
        )
