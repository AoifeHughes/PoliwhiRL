# -*- coding: utf-8 -*-
from model import run
import torch
import os

def main():
    rom_path = "Pokemon - Crystal Version.gbc"
    device = torch.device("mps")
    SCALE_FACTOR = 1
    USE_GRAYSCALE = False
    timeouts = [1000]
    num_episodes = 1000
    cpus = (os.cpu_count()-1) if device == torch.device("cpu") else 1
    episodes_per_batch = 1*cpus
    nsteps = 5
    batch_size = timeouts[0]//nsteps
    run(rom_path, device, SCALE_FACTOR, USE_GRAYSCALE, timeouts, num_episodes, episodes_per_batch, batch_size, nsteps, cpus=cpus)


if __name__ == "__main__":
    main()
