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
        x = torch.squeeze(x, 1)  # Squeeze the tensor to remove the extra dimension
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
        x = torch.squeeze(x, 1)  # Squeeze the tensor to remove the extra dimension
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)



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
    device,
    epsilon=0.2,
):
    # Make sure tensors are on the correct device
    states = states.to(device)
    actions = actions.to(device)
    log_probs_old = log_probs_old.to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)

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
    return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}


class PPOBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
        self.returns = []
        self.advantages = []

    def add(self, state, action, log_prob, reward, value, is_terminal):
        self.states.append(state.clone().detach())
        self.actions.append(action.clone().detach())
        self.log_probs.append(log_prob.clone().detach())
        self.rewards.append(reward)
        self.values.append(value.clone().detach())
        self.is_terminals.append(is_terminal)

    def compute_gae_and_returns(self, last_value, gamma=0.99, tau=0.95):
        gae = 0
        self.returns = []  # Reset returns to avoid accumulation during multiple calls
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * last_value * (1 - self.is_terminals[step]) - self.values[step]
            gae = delta + gamma * tau * (1 - self.is_terminals[step]) * gae
            self.returns.insert(0, gae + self.values[step])
            last_value = self.values[step]
        self.returns = torch.tensor(self.returns, dtype=torch.float32)
        self.advantages = self.returns - torch.tensor(self.values, dtype=torch.float32)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batch(self):
        """Concatenate all lists into tensors."""
        states_tensor = torch.stack(self.states)
        actions_tensor = torch.stack(self.actions).squeeze().to(torch.int64)
        log_probs_tensor = torch.stack(self.log_probs)
        returns_tensor = torch.tensor(self.returns, dtype=torch.float32)
        advantages_tensor = torch.tensor(self.advantages, dtype=torch.float32)
        return states_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages_tensor

    def merge(self, other_buffer):
        """Merge another PPOBuffer into this one."""
        self.states.extend(other_buffer.states)
        self.actions.extend(other_buffer.actions)
        self.log_probs.extend(other_buffer.log_probs)
        self.rewards.extend(other_buffer.rewards)
        self.values.extend(other_buffer.values)
        self.is_terminals.extend(other_buffer.is_terminals)
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
        self.states = []  # List to collect state tensors
        self.actions = []  # List to collect action tensors
        self.log_probs = []  # List to collect log probability tensors
        self.rewards = []  # List to collect scalar rewards
        self.values = []  # List to collect value tensors
        self.is_terminals = []  # List to collect terminal flags (scalars)
        self.returns = []  # Reset or could also initialize an empty tensor if needed
        self.advantages = []  # Reset or could also initialize an empty tensor if needed

    def add(self, state, action, log_prob, reward, value, is_terminal):
        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward)  # Assuming reward is a scalar or already detached
        self.values.append(value.detach().cpu())
        self.is_terminals.append(is_terminal)


    def compute_gae_and_returns(self, next_value, gamma=0.99, tau=0.95):
        gae = 0
        self.returns = []  # Reset returns to avoid accumulation during multiple calls
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * next_value * (1 - self.is_terminals[step]) - self.values[step]
            gae = delta + gamma * tau * (1 - self.is_terminals[step]) * gae
            self.returns.insert(0, gae + self.values[step])
            next_value = self.values[step]
        self.returns = torch.tensor(self.returns, dtype=torch.float32).detach()
        self.advantages = self.returns - torch.tensor(self.values, dtype=torch.float32).detach()

    def get_batch(self):
        # Convert to tensors
        states_tensor = torch.stack(self.states)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long)
        log_probs_tensor = torch.stack(self.log_probs)
        returns_tensor = self.returns
        advantages_tensor = self.advantages
        is_terminals_tensor = torch.tensor(self.is_terminals, dtype=torch.float32)

        self.clear()  # Clear buffer after getting the batch

        return states_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages_tensor, is_terminals_tensor

    def merge(self, buffer):
        """Merge other PPOBuffers into this one. Useful for collecting data from multiple processes."""
        self.states.extend(buffer.states)
        self.actions.extend(buffer.actions)
        self.log_probs.extend(buffer.log_probs)
        self.rewards.extend(buffer.rewards)
        self.values.extend(buffer.values)
        self.is_terminals.extend(buffer.is_terminals)