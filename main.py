# -*- coding: utf-8 -*-
import DQN
import torch
import memory


def main():
    rom_path = "Pokemon - Crystal Version.gbc"

    location_address = memory.location
    locations = memory.locations
    device = torch.device("mps")
    SCALE_FACTOR = 0.5
    USE_GRAYSCALE = True
    goal_locs = [locations[6], locations[4]]
    timeout = 300
    goal_targets = [400, 400]
    model = DQN.LearnGame(
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
    model.run()


if __name__ == "__main__":
    main()
