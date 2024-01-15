# -*- coding: utf-8 -*-
from model import run
import torch


def main():
    rom_path = "Pokemon - Crystal Version.gbc"
    device = torch.device("cpu")
    SCALE_FACTOR = 0.5
    USE_GRAYSCALE = True
    playtime = 1 #1 hour and each move is about 1 second
    timeout = 60 * 60 * playtime
    num_episodes = 500
    run(rom_path,  device, SCALE_FACTOR, USE_GRAYSCALE, timeout, num_episodes)


if __name__ == "__main__":
    main()
