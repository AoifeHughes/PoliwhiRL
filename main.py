# -*- coding: utf-8 -*-
from PoliwhiRL.models.DQN.DQN_run import run
import torch
import os


def main():
    rom_path = "./emu_files/Pokemon - Crystal Version.gbc"
    device = torch.device("mps")
    SCALE_FACTOR = 0.5
    USE_GRAYSCALE = True
    timeouts = [15] 
    state_paths = [
        "./emu_files/states/start.state",
        #"./states/outside.state",
        #"./states/lab.state",
        #"./states/battle.state",
    ]
    cpus = (os.cpu_count() - 1) if device == torch.device("cpu") else 1
    episodes_per_batch = 2 * cpus
    num_episodes = 5000 * cpus
    nsteps = 3
    batch_size = 16
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
