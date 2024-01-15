# -*- coding: utf-8 -*-
import DQN
import torch
import memory


def main():
    rom_path = "Pokemon - Crystal Version.gbc"

    location_address = memory.location
    locations = memory.locations
    device = torch.device("cpu")
    SCALE_FACTOR = 0.5
    USE_GRAYSCALE = False
    goal_locs = [locations[6], locations[4]]
    timeout = 3
    goal_targets = [300, 600]
    DQN.run_model(
        rom_path,
        locations,
        location_address,
        device,
        SCALE_FACTOR,
        USE_GRAYSCALE,
        goal_locs,
        goal_targets,
        timeout,
    )


if __name__ == "__main__":
    main()
