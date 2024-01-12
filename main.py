import DQN
import torch
import memory 
def main():
    rom_path = 'Pokemon - Crystal Version.gbc'
    locations = {6: "DownstairsPlayersHouse", 0: "UpstairsPlayersHouse", 4: "OutsideStartingArea"}
    location_address = 0xD148
    device = torch.device("mps")
    SCALE_FACTOR = 0.25
    USE_GRAYSCALE = True
    goal_loc = memory.outside_house
    model = DQN.LearnGame(rom_path, locations, location_address, device, SCALE_FACTOR, USE_GRAYSCALE, goal_loc)
    model.run()

if __name__ == '__main__':
    main()