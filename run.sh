#!/bin/bash

base_episode_length=256
num_workers=8
runs_per_worker=4


for ((i=1; i<=5; i++))
do
    # Calculate episode_length for the current iteration
    episode_length=$((base_episode_length * i))

    # Check if the directory already exists
    if [ -d "run_$i" ]; then
        echo "Directory run_$i already exists, skipping..."
        continue
    fi

    # Execute the Python script with dynamic episode_length
    python main.py --use_config ./configs/multi_config.json --episode_length $((base_episode_length * i)) --num_workers $num_workers --runs_per_worker $runs_per_worker 

    # Create directory and move files
    mkdir run_$i
    mv results run_$i
    mv runs/* run_$i
    mv logs run_$i
    mv checkpoints run_$i
done
