import numpy as np
from multiprocessing import Manager

class ReplayMemory(object):
    def __init__(self, capacity, n_steps=5, multiCPU=True):
        manager = Manager()
        self.memory = manager.list() if multiCPU else []
        self.lock = manager.Lock() if multiCPU else DummyLock()
        self.capacity = capacity
        self.n_steps = n_steps

    def push(self, *args):
        """Saves a transition."""
        with self.lock:
            self.memory.append(args)
            if len(self.memory) > self.capacity:
                self.memory.pop(0)          
    def sample(self, batch_size):
        with self.lock:
            return [
                self.memory[i]
                for i in np.random.choice(
                    np.arange(len(self.memory)), batch_size, replace=False
                )
            ]

    def __len__(self):
        with self.lock:
            return len(self.memory)


class DummyLock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
