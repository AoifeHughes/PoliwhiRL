# -*- coding: utf-8 -*-

from PoliwhiRL.models.RainbowDQN.RainbowDQN import run as run_rainbow
from PoliwhiRL.models.DQN.DQN import run as run_dqn
from torch import device
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom_path", type=str, default="./emu_files/Pokemon - Crystal Version.gbc")
    parser.add_argument("--state_path", type=str, default="./emu_files/states/start.state")
    parser.add_argument("--episode_length", type=int, default=5)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/RainbowDQN.pth")
    parser.add_argument("--model", type=str, default="RainbowDQN")
    return parser.parse_args()

def main():

    args = parse_args()
    rom_path = args.rom_path
    state_path = args.state_path
    episode_length = args.episode_length
    d = device(args.device)
    num_episodes = args.num_episodes
    batch_size = args.batch_size
    checkpoint = args.checkpoint

    if args.model == "RainbowDQN":
        run_rainbow(rom_path, state_path, episode_length, d, num_episodes, batch_size, checkpoint)
    elif args.model == "DQN":
        raise NotImplementedError
    elif args.model == "PPO":
        raise NotImplementedError
    else:
        raise ValueError(f"Model {args.model} not recognized")

if __name__ == "__main__":
    main()
