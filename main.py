# -*- coding: utf-8 -*-

from PoliwhiRL.models.RainbowDQN import run as run_rainbow
from PoliwhiRL.models.RainbowDQN.doubleRainbow import run as run_rainbow_parallel
from PoliwhiRL.models.DQN.DQN import run as run_dqn
from PoliwhiRL.environment.explore import explore
from torch import device
import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rom_path", type=str, default="./emu_files/Pokemon - Crystal Version.gbc"
    )
    parser.add_argument(
        "--state_path", type=str, default="./emu_files/states/start.state"
    )
    parser.add_argument("--episode_length", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/RainbowDQN.pth"
    )
    parser.add_argument("--model", type=str, default="RainbowDQN")
    parser.add_argument("--sight", action="store_true")
    parser.add_argument("--erase", action="store_true")
    parser.add_argument("--parallel", action="store_true")
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
    sight = args.sight
    parallel = True
    erase = True


    if erase:
        print("Erasing all logs, checkpoints, runs, and results")
        folders = ["checkpoints", "logs", "runs", "results"]
        for f in folders:
            if f in os.listdir():
                shutil.rmtree(f)

    if args.model == "RainbowDQN":
        if parallel:
            run_rainbow_parallel(
                rom_path,
                state_path,
                episode_length,
                d,
                num_episodes,
                batch_size,
                checkpoint,
                sight)
        else:
            run_rainbow(
                rom_path,
                state_path,
                episode_length,
                d,
                num_episodes,
                batch_size,
                checkpoint,
                sight,
                parallel,
            )
    elif args.model == "DQN":
        raise NotImplementedError
    elif args.model == "PPO":
        raise NotImplementedError
    elif args.model == "explore":
        explore(num_episodes, rom_path, state_path, episode_length, sight)
    else:
        raise ValueError(f"Model {args.model} not recognized")


if __name__ == "__main__":
    main()
