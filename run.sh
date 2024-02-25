#!/bin/bash

base_episode_length=256
num_workers=8
runs_per_worker=1

# Array to hold grayscale usage flags
use_grayscale_flags=("true" "false")

# Outer loop to toggle grayscale flag
for use_grayscale in "${use_grayscale_flags[@]}"
do
    for ((i=1; i<=5; i++))
    do
        # Calculate episode_length for the current iteration
        episode_length=$((base_episode_length * i))

        # Check if the directory already exists
        if [ -d "run_${i}_grayscale_${use_grayscale}" ]; then
            echo "Directory run_${i}_grayscale_${use_grayscale} already exists, skipping..."
            continue
        fi

        # Execute the Python script with dynamic episode_length and grayscale flag
        python main.py --use_config ./configs/multi_config.json --episode_length $episode_length --num_workers $num_workers --runs_per_worker $runs_per_worker --use_grayscale $use_grayscale

        # Create directory and move files
        mkdir "run_${i}_grayscale_${use_grayscale}"
        mv results "run_${i}_grayscale_${use_grayscale}"
        mv runs/* "run_${i}_grayscale_${use_grayscale}"
        mv logs "run_${i}_grayscale_${use_grayscale}"
        mv checkpoints "run_${i}_grayscale_${use_grayscale}"
    done
done
