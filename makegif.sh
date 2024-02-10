#!/bin/bash

# Check for proper number of command line args.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 images_directory output_gif"
    exit 1
fi

# Command line arguments
images_directory="$1"
output_gif="$2"

# Frame rate (5 fps)
fps=5

# Create a temporary palette for better quality
palette="/tmp/palette.png"

# Generate palette
ffmpeg -i "${images_directory}/%*.jpg" -vf "fps=$fps,scale=320:-1:flags=lanczos,palettegen" -y $palette

# Create GIF using ffmpeg and the generated palette
ffmpeg -i "${images_directory}/%*.jpg" -i $palette -filter_complex "fps=$fps,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse" -y $output_gif

echo "GIF created: $output_gif"
