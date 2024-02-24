#!/bin/bash

# Define variables
device="mps"
model="RainbowDQN"
checkpoint_interval=1000
epsilon_by_location="--epsilon_by_location" # This is a flag
num_episodes=10000
base_episode_length=100 # Base episode length before multiplication
scale_factor=0.25

for ((i=1; i<=1; i++))
do
    # Calculate episode_length for the current iteration
    episode_length=$((base_episode_length * i))

    # Check if the directory already exists
    if [ -d "run_$i" ]; then
        echo "Directory run_$i already exists, skipping..."
        continue
    fi

    # Execute the Python script with dynamic episode_length
    python main.py --device $device --model $model --checkpoint_interval $checkpoint_interval $epsilon_by_location --num_episodes $num_episodes --episode_length $episode_length --scaling_factor $scale_factor

    # Create directory and move files
    mkdir run_$i
    mv results run_$i
    mv runs/* run_$i
    mv logs run_$i
done
