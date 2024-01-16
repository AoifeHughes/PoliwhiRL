# -*- coding: utf-8 -*-
from model import run
import torch


def main():
    rom_path = "Pokemon - Crystal Version.gbc"
    device = torch.device("cpu")
    SCALE_FACTOR = 0.5
    USE_GRAYSCALE = True
    timeouts = [5, 10]
    num_episodes = 130
    run(rom_path,  device, SCALE_FACTOR, USE_GRAYSCALE, timeouts, num_episodes)


if __name__ == "__main__":
    main()
