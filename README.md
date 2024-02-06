# PoliwhiRL README

![PoliwhiRL](./images/logo.png)

## Overview
PoliwhiRL is a Reinforcement Learning library designed for sprite-based 2-D Pokémon games. It provides a framework for experimenting with different RL algorithms to interact with Pokémon games in an emulated environment.

## Command-Line Options
The main script `main.py` supports several command-line options to customize the execution of the RL models. Here are the options available:

- `--rom_path`: Path to the ROM file. Default is `./emu_files/Pokemon - Crystal Version.gbc`.
- `--state_path`: Path to the saved state file for starting the game. Default is `./emu_files/states/start.state`.
- `--episode_length`: Length of each episode in terms of the number of steps. Default is 5.
- `--device`: The device to run the model on (`cpu` `mps`, `cuda`). Default is `cpu`.
- `--num_episodes`: Number of episodes to run. Default is 1000.
- `--batch_size`: Batch size for training. Default is 32.
- `--checkpoint`: Path to save or load the model checkpoint. Default is `./checkpoints/RainbowDQN.pth`.
- `--model`: The model to use. Options are `RainbowDQN`, `DQN`, `PPO`, or `explore`. Default is `RainbowDQN`.
- `--sight`: Enable sight-based exploration.
- `--erase`: Erase all logs, checkpoints, runs, and results before starting.
- `--parallel`: Run the model in parallel mode.
- `--runs_per_worker`: Number of runs per worker in parallel mode. Default is 4.
- `--num_workers`: Number of workers for parallel execution. Default is 8.

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

For more details on the implementation and usage, refer to the [PoliwhiRL GitHub repository](https://github.com/AoifeHughes/PoliwhiRL