#!/bin/bash

# Define the base command to run
command="python main.py --model explore --episode_length 10000 --num_episodes 1"

# Directory containing state files
state_dir="./emu_files/states/"

# Maximum number of instances to run concurrently
max_instances=1

# Function to run the command with optional 'sight' and the specified 'state_path'
run_command() {
    sight=$1
    state_file=$2
    start_time=$(date +%s)

    # Check if sight is enabled and adjust the command accordingly
    if [ "$sight" = "True" ]; then
        final_command="$command --sight --state_path \"$state_file\""
    else
        final_command="$command --state_path \"$state_file\""
    fi

    # Execute the final command
    eval $final_command

    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    echo "Execution with 'sight=$sight' and 'state_path=$state_file' completed in $execution_time seconds"
}

# Ensure state directory exists
if [ ! -d "$state_dir" ]; then
    echo "State directory '$state_dir' not found."
    exit 1
fi

# Get a list of state files in the directory
state_files=("$state_dir"/*)

# Loop through the state files and run commands in parallel with limited instances
instance_count=0
for state_file in "${state_files[@]}"; do
    if [ $instance_count -ge $max_instances ]; then
        wait
        instance_count=0
    fi

    # Run commands with and without sight in parallel
    run_command True "$state_file" &
    run_command False "$state_file" &
    ((instance_count+=2)) # Increment by 2 because we are launching two instances per loop

    # Ensure we don't exceed max instances
    if [ $instance_count -ge $max_instances ]; then
        wait
        instance_count=0
    fi
done

# Wait for any remaining background processes to finish
wait

echo "All commands have finished."
