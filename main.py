# -*- coding: utf-8 -*-

from PoliwhiRL.models.RainbowDQN import run as rainbow
from PoliwhiRL.environment import explore
from torch import device
import os
import shutil
import argparse
import json


def load_default_config():
    default_config_path = './configs/default_config.json'  # Path to the default config file
    if os.path.exists(default_config_path):
        with open(default_config_path, "r") as f:
            return json.load(f)
    return {}


def load_user_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def merge_configs(default_config, user_config):
    merged_config = default_config.copy()
    merged_config.update(user_config)
    return merged_config


def parse_args():
    default_config = load_default_config()
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_config", type=str, default=None, help="Path to user config file")
    args, unknown = parser.parse_known_args()  # Parse known args first to get config file if specified

    user_config = load_user_config(args.use_config)
    config = merge_configs(default_config, user_config)

    # Add other arguments
    parser.add_argument("--rom_path", type=str, default=config.get("rom_path"))
    parser.add_argument("--scaling_factor", type=float, default=config.get("scaling_factor"))
    parser.add_argument("--state_path", type=str, default=config.get("state_path"))
    parser.add_argument("--episode_length", type=int, default=config.get("episode_length"))
    parser.add_argument("--device", type=str, default=config.get("device"))
    parser.add_argument("--num_episodes", type=int, default=config.get("num_episodes"))
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size"))
    parser.add_argument("--checkpoint", type=str, default=config.get("checkpoint"))
    parser.add_argument("--model", type=str, default=config.get("model"))
    parser.add_argument("--sight", action="store_true", default=config.get("sight"))
    parser.add_argument("--erase", action="store_true", default=config.get("erase"))
    parser.add_argument("--parallel", action="store_true", default=config.get("parallel"))
    parser.add_argument("--runs_per_worker", type=int, default=config.get("runs_per_worker"))
    parser.add_argument("--num_workers", type=int, default=config.get("num_workers"))
    parser.add_argument("--checkpoint_interval", type=int, default=config.get("checkpoint_interval"))
    parser.add_argument("--epsilon_by_location", action="store_true", default=config.get("epsilon_by_location"))
    parser.add_argument("--extra_files", type=json.loads, default=json.dumps(config.get("extra_files")))
    parser.add_argument("--reward_locations_xy", type=json.loads, default=json.dumps(config.get("reward_locations_xy")))
    parser.add_argument("--use_grayscale", action="store_true", default=config.get("use_grayscale"))

    return parser.parse_args()


def main():
    args = parse_args()

    if args.erase:
        print("Erasing all logs, checkpoints, runs, and results")
        folders = ["checkpoints", "logs", "runs", "results"]
        for f in folders:
            if f in os.listdir():
                shutil.rmtree(f)

    d = device(args.device)

    if args.model == "RainbowDQN":
        if args.parallel and d != device("cpu"):
            print("Parallel RainbowDQN only supports CPU devices. Switching to CPU.")
            d = device("cpu")
        rainbow(
            args.rom_path,
            args.state_path,
            args.episode_length,
            d,
            args.num_episodes,
            args.batch_size,
            args.checkpoint,
            args.parallel,
            args.sight,
            args.runs_per_worker,
            args.num_workers,
            args.checkpoint_interval,
            args.epsilon_by_location,
            args.extra_files,
            args.reward_locations_xy,
            args.scaling_factor,
            args.use_grayscale,
        )
    elif args.model in ["DQN", "PPO"]:
        raise NotImplementedError(f"{args.model} is not implemented yet.")
    elif args.model == "explore":
        explore(args.num_episodes, args.rom_path, args.state_path, args.episode_length, args.sight)
    else:
        raise ValueError(f"Model {args.model} not recognized")


if __name__ == "__main__":
    main()
