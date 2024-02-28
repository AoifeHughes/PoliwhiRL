#!/bin/bash

base_episode_length=64
num_workers=6
runs_per_worker=1

# Array to hold grayscale usage flags
use_grayscale_flags=("true" "false")
run_parallel_flags=("true" "false")
# Array for scaling_factor values
scaling_factors=(0.25 0.5 1)

for run_parallel in "${run_parallel_flags[@]}"
do
    # Outermost loop for scaling_factor
    for scaling_factor in "${scaling_factors[@]}"
    do
        # Outer loop to toggle grayscale flag
        for use_grayscale in "${use_grayscale_flags[@]}"
        do
            for ((i=1; i<=5; i++))
            do
                # Calculate episode_length for the current iteration, adjusted by scaling_factor
                episode_length=$(echo "$base_episode_length * $i * $scaling_factor" | bc)
                episode_length=${episode_length%.*}  # Convert to integer if necessary

                # Define a directory name that includes scaling_factor
                dir_name="run_${i}_scale_${scaling_factor}_grayscale_${use_grayscale}_parallel_${run_parallel}"

                # Check if the directory already exists
                if [ -d "$dir_name" ]; then
                    echo "Directory $dir_name already exists, skipping..."
                    continue
                fi

                # Execute the Python script with dynamic episode_length, grayscale flag, and scaling factor
                python main.py --use_config ./configs/multi_config.json --episode_length $episode_length --num_workers $num_workers --runs_per_worker $runs_per_worker --use_grayscale $use_grayscale --scaling_factor $scaling_factor --run_parallel $run_parallel 

                # Create directory and move files
                mkdir "$dir_name"
                mv results "$dir_name"
                mv runs/* "$dir_name"
                mv logs "$dir_name"
                rm -r checkpoints
            done
        done
    done
done
