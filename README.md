# PoliwhiRL README
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/AoifeHughes/PoliwhiRL/main.svg)](https://results.pre-commit.ci/latest/github/AoifeHughes/PoliwhiRL/main)

<img src="./images/logo.png" alt="PoliwhiRL" width="50%">

## Legal Disclaimer
This project is not affiliated with, endorsed, or sponsored by Nintendo, Game
Freak, or The Pokémon Company. All Pokémon games and characters are trademarks
of Nintendo, Game Freak, and The Pokémon Company. This project is intended for
educational and research purposes only.

All usage of The Pokemon Company International's games is done with the
understanding that software is legally obtained and that the user has the right to use it. This project does not provide any ROMs or copyrighted materials.

## Overview
PoliwhiRL is a Reinforcement Learning library designed for sprite-based 2-D Pokémon games. It provides a framework for experimenting with different RL algorithms to interact with Pokémon games in an emulated environment.

## Command-Line Options

The main script `main.py` supports several command-line options to customize the execution of the RL models. Below is a list of the available options along with their default values, which can be overridden by specifying a configuration file with `--use_config` or directly via command-line arguments:

- `--use_config`: Path to a user-specified configuration JSON file. This allows you to load all the settings from a file, overriding the defaults specified below.

- `--rom_path`: Path to the ROM file. Default is `./emu_files/Pokemon - Crystal Version.gbc`.
- `--state_path`: Path to the saved state file for starting the game. Default is `./emu_files/states/start.state`.
- `--episode_length`: Length of each episode in terms of the number of steps. Default is 25.
- `--device`: The device to run the model on (`cpu`, `mps`, `cuda`). Default is `cpu`.
- `--num_episodes`: Number of episodes to run. Default is 10000.
- `--batch_size`: Batch size for training. Default is 128 (Note: This is adjusted to match the provided configuration).
- `--checkpoint`: Path to save or load the model checkpoint. Default is `./checkpoints/RainbowDQN.pth`.
- `--model`: The model to use. Options are `RainbowDQN`, `DQN`, `PPO`, or `explore`. Default is `RainbowDQN`.
- `--sight`: Enable sight-based exploration. Default is disabled.
- `--erase`: Erase all logs, checkpoints, runs, and results before starting. Default is disabled.
- `--parallel`: Run the model in parallel mode. Default is disabled.
- `--runs_per_worker`: Number of runs per worker in parallel mode. Default is 4.
- `--num_workers`: Number of workers for parallel execution. Default is 6.
- `--checkpoint_interval`: Interval for saving the model checkpoint. Default is 100.
- `--epsilon_by_location`: Enable epsilon decay by location strategy. Default is disabled.
- `--extra_files`: List of extra files to use. Default includes additional RAM and RTC files for the emulator.
- `--reward_locations_xy`: Dictionary of XY locations to rewards, specified as a JSON string. Default includes several predefined locations.
- `--use_grayscale`: Enable the use of grayscale images. Default is disabled.

These options provide flexibility in configuring the environment and model parameters for training and evaluating reinforcement learning models.


## Configuration via `config.json`

In addition to command-line options, you can also customize the execution of the RL models by editing the `config.json` file. This file allows you to set default values for all the parameters available in command-line options. If `config.json` exists in the same directory as `main.py`, it will be automatically loaded, and its settings will be used as defaults, which can still be overridden by command-line arguments.

This approach provides a flexible way to maintain different configurations without changing the script or specifying many command-line arguments every time you run the model.
## Implemented Features
- **Models**: RainbowDQN, DQN (partially implemented), and PPO (planned but not implemented).
- **Exploration**: Basic exploration functionality with optional sight-based exploration.
- **Environment Interaction**: Control and interaction with the game environment, including image-based memory for tracking visited states.
- **Utilities**: OCR for text extraction from images and utility functions for various tasks.

## To Be Finished
- **DQN and PPO Models**: The DQN model is partially implemented, and the PPO model is planned but not yet implemented.
- **Parallel Exploration**: The framework for parallel exploration is outlined but not fully implemented.
- **Comprehensive Documentation**: Detailed documentation for each component and model within the library is needed.
- **Extended Game Support**: Currently, the library is tailored for Pokémon Crystal. Expanding support for other sprite-based 2-D Pokémon games would enhance its utility.

## Requirements
The library dependencies are specified in `requirements.txt`, including PyTorch, NumPy, Matplotlib, and others necessary for running the models and processing the game environment.

For more details on the implementation and usage, refer to the [PoliwhiRL GitHub
repository](https://github.com/AoifeHughes/PoliwhiRL)

## Paper
For more details on the implementation and the results, refer to the [PoliwhiRL paper](./paper/paper.pdf).