# -*- coding: utf-8 -*-
import DQN
import torch
import memory


def main():
    rom_path = "Pokemon - Crystal Version.gbc"

    location_address = memory.location
    locations = memory.locations
    device = torch.device("mps")
    SCALE_FACTOR = 1
    USE_GRAYSCALE = True
    goal_loc = locations[6]
    timeout = -1
    model = DQN.LearnGame(
        rom_path,
        locations,
        location_address,
        device,
        SCALE_FACTOR,
        USE_GRAYSCALE,
        goal_loc,
        timeout,
    )
    model.run()


if __name__ == "__main__":
    main()
