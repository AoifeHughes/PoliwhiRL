#!/bin/bash

# Directory containing state files
state_files_dir="./emu_files/states"

# Loop over each file in the states directory
for state_file in "$state_files_dir"/*; do
    # Check if the file is a regular file (not a directory or a link)
    if [ -f "$state_file" ]; then
        # Run the python command with the current state file
        echo "Running command with state file: $state_file"
        python main.py --model explore --episode_length 100000 --num_episodes 1 --state_path "$state_file"
    fi
done
