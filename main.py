# -*- coding: utf-8 -*-

from PoliwhiRL.models.RainbowDQN import run as rainbow
from PoliwhiRL.environment import explore
from torch import device
import os
import shutil
import argparse
import json


def parse_args():
    # Load default configuration from file if exists
    default_config = "config.json"
    config = {}
    if os.path.exists(default_config):
        with open(default_config, "r") as f:
            config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rom_path",
        type=str,
        default=config.get("rom_path", "./emu_files/Pokemon - Crystal Version.gbc"),
    )
    parser.add_argument(
        "--scaling_factor", type=float, default=config.get("scaling_factor", 1)
    )
    parser.add_argument(
        "--state_path",
        type=str,
        default=config.get("state_path", "./emu_files/states/start.state"),
    )
    parser.add_argument(
        "--episode_length", type=int, default=config.get("episode_length", 25)
    )
    parser.add_argument("--device", type=str, default=config.get("device", "cpu"))
    parser.add_argument(
        "--num_episodes", type=int, default=config.get("num_episodes", 10000)
    )
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 32))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=config.get("checkpoint", "./checkpoints/RainbowDQN.pth"),
    )
    parser.add_argument("--model", type=str, default=config.get("model", "RainbowDQN"))
    parser.add_argument(
        "--sight", action="store_true", default=config.get("sight", False)
    )
    parser.add_argument(
        "--erase", action="store_true", default=config.get("erase", False)
    )
    parser.add_argument(
        "--parallel", action="store_true", default=config.get("parallel", False)
    )
    parser.add_argument(
        "--runs_per_worker", type=int, default=config.get("runs_per_worker", 4)
    )
    parser.add_argument("--num_workers", type=int, default=config.get("num_workers", 6))
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=config.get("checkpoint_interval", 100),
    )
    parser.add_argument(
        "--epsilon_by_location",
        action="store_true",
        default=config.get("epsilon_by_location", False),
    )
    parser.add_argument(
        "--extra_files", type=list, default=config.get("extra_files", [])
    )
    parser.add_argument(
        "--reward_locations_xy",
        type=json.loads,
        default=config.get("reward_locations_xy", "{}"),
    )

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
    parallel = args.parallel
    erase = args.erase
    runs_per_worker = args.runs_per_worker
    num_workers = args.num_workers
    checkpoint_interval = args.checkpoint_interval
    epsilon_by_location = args.epsilon_by_location
    extra_files = args.extra_files
    reward_locations_xy = {int(k): v for k, v in args.reward_locations_xy.items()}
    scaling_factor = args.scaling_factor

    if erase:
        print("Erasing all logs, checkpoints, runs, and results")
        folders = ["checkpoints", "logs", "runs", "results"]
        for f in folders:
            if f in os.listdir():
                shutil.rmtree(f)

    if args.model == "RainbowDQN":
        if parallel:
            if d != device("cpu"):
                print(
                    "Parallel RainbowDQN only supports CPU devices. Switching to CPU."
                )
                d = device("cpu")
        rainbow(
            rom_path,
            state_path,
            episode_length,
            d,
            num_episodes,
            batch_size,
            checkpoint,
            parallel,
            sight,
            runs_per_worker,
            num_workers,
            0,
            checkpoint_interval,
            epsilon_by_location,
            extra_files,
            reward_locations_xy,
            scaling_factor,
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
