# -*- coding: utf-8 -*-
from learn import run
import torch
import os


def main():
    rom_path = "Pokemon - Crystal Version.gbc"
    device = torch.device("cpu")
    SCALE_FACTOR = 1
    USE_GRAYSCALE = False
    timeouts = [10, 15, 20, 50, 100, 1000]
    state_paths = [
        "./states/start.state",
        "./states/outside.state",
        "./states/lab.state",
        "./states/battle.state",
    ]
    cpus = (os.cpu_count() - 1) if device == torch.device("cpu") else 1
    episodes_per_batch = 2 * cpus
    num_episodes = 50*cpus
    nsteps = 3
    batch_size = 32
    explore_mode = False
    run(
        rom_path,
        device,
        SCALE_FACTOR,
        USE_GRAYSCALE,
        timeouts,
        state_paths[::-1],
        num_episodes,
        episodes_per_batch,
        batch_size,
        nsteps,
        cpus=cpus,
        explore_mode=explore_mode,
    )

if __name__ == "__main__":
    main()
