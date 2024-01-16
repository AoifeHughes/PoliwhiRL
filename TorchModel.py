
import torch.nn as nn
import torch 
from collections import deque
import random
from multiprocessing import Lock, Manager


class DQN(nn.Module):
    def __init__(self, h, w, outputs, USE_GRAYSCALE):
        super(DQN, self).__init__()
        self.USE_GRAYSCALE = USE_GRAYSCALE
        self.conv1 = nn.Conv2d(1 if USE_GRAYSCALE else 3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self._to_linear = None
        self._compute_conv_output_size(h, w)
        self.fc = nn.Linear(self._to_linear, outputs)

    def _compute_conv_output_size(self, h, w):
        x = torch.rand(1, 1 if self.USE_GRAYSCALE else 3, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ReplayMemory(object):
    def __init__(self, capacity):
        manager = Manager()
        self.memory = manager.list()
        self.lock = manager.Lock()
        self.capacity = capacity

    def push(self, *args):
        with self.lock:
            # Ensure memory does not exceed capacity
            if len(self.memory) >= self.capacity:
                self.memory.pop(0)
            self.memory.append(args)

    def sample(self, batch_size):
        with self.lock:
            memory_length = len(self.memory)
            indices = random.sample(range(memory_length), min(memory_length, batch_size))
            return [self.memory[i] for i in indices]

    def __len__(self):
        with self.lock:
            return len(self.memory)
