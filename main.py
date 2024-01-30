# -*- coding: utf-8 -*-
from learn import run
import torch
import os


def main():
    rom_path = "Pokemon - Crystal Version.gbc"
    device = torch.device("mps")
    SCALE_FACTOR = 1
    USE_GRAYSCALE = False
    timeouts = [25]
    state_paths = [
        "./states/start.state",
        "./states/outside.state",
        "./states/lab.state",
    ]
    num_episodes = 100
    cpus = (os.cpu_count() - 1) if device == torch.device("cpu") else 1
    episodes_per_batch = 1 * cpus
    nsteps = 5
    batch_size = timeouts[0] // nsteps
    explore_mode = False
    run(
        rom_path,
        device,
        SCALE_FACTOR,
        USE_GRAYSCALE,
        timeouts,
        state_paths,
        num_episodes,
        episodes_per_batch,
        batch_size,
        nsteps,
        cpus=cpus,
        explore_mode=explore_mode,
    )


if __name__ == "__main__":
    main()
