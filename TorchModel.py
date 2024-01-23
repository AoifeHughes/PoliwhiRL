
import torch.nn as nn
import torch 
from collections import deque
import random
from multiprocessing import Lock, Manager
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, h, w, outputs, USE_GRAYSCALE):
        super(DQN, self).__init__()
        self.USE_GRAYSCALE = USE_GRAYSCALE
        # Convolutional layers
        self.conv1 = nn.Conv2d(1 if USE_GRAYSCALE else 3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)  # Additional convolutional layer
        self.bn4 = nn.BatchNorm2d(64)
        
        self._to_linear = None
        self._compute_conv_output_size(h, w)
        self.fc1 = nn.Linear(self._to_linear, 512)  # Larger fully connected layer
        self.fc2 = nn.Linear(512, outputs)          # Additional fully connected layer
        self.dropout = nn.Dropout(0.5)              # Dropout layer

    def _compute_conv_output_size(self, h, w):
        x = torch.rand(1, 1 if self.USE_GRAYSCALE else 3, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ReplayMemory(object):
    def __init__(self, capacity, n_steps=5):
        manager = Manager()
        self.memory = manager.list()
        self.lock = manager.Lock()
        self.capacity = capacity
        self.n_steps = n_steps
        self.temporal_buffer = []

    def push(self, *args):
        # Add transition to the temporal buffer
        self.temporal_buffer.append(args)

        if len(self.temporal_buffer) == self.n_steps:
            with self.lock:
                # Ensure memory does not exceed capacity
                if len(self.memory) >= self.capacity:
                    # pop a random element to make space
                    self.memory.pop(random.randrange(len(self.memory)))
                # Push the sequence of N steps
                self.memory.append(list(self.temporal_buffer))
                # Reset the temporal buffer
                self.temporal_buffer = []

    def sample(self, batch_size):
        with self.lock:
            memory_length = len(self.memory)
            indices = random.sample(range(memory_length), min(memory_length, batch_size))
            return [self.memory[i] for i in indices]

    def __len__(self):
        with self.lock:
            return len(self.memory)
