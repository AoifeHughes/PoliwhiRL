# -*- coding: utf-8 -*-
from model import run
import torch
import os

def main():
    rom_path = "Pokemon - Crystal Version.gbc"
    device = torch.device("mps")
    SCALE_FACTOR = 1
    USE_GRAYSCALE = False
    timeouts = [10, 50, 100, 500, 1000]
    num_episodes = 100
    cpus = (os.cpu_count()-1) if device == torch.device("cpu") else 1
    episodes_per_batch = 1*cpus
    batch_size = 16
    nsteps = 3
    run(rom_path, device, SCALE_FACTOR, USE_GRAYSCALE, timeouts, num_episodes, episodes_per_batch, batch_size, nsteps, cpus=cpus)


if __name__ == "__main__":
    main()
