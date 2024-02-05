# -*- coding: utf-8 -*-

from PoliwhiRL.models.RainbowDQN.RainbowDQN import run
from torch import device 


def main():

    rom_path = "./emu_files/Pokemon - Crystal Version.gbc"
    state_path = "./emu_files/states/start.state"
    episode_length = 10
    d = device("mps")
    num_episodes = 10000
    batch_size = 32
    checkpoint = "./checkpoints/RainbowDQN.pth"
    run(rom_path, state_path, episode_length, d, num_episodes, batch_size, checkpoint)


if __name__ == "__main__":
    main()
