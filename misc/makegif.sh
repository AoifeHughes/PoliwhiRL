#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder_location> <output_filename.gif>"
    exit 1
fi

FOLDER_LOCATION=$1
OUTPUT_FILENAME=$2

if [ ! -d "$FOLDER_LOCATION" ]; then
    echo "The specified folder does not exist."
    exit 1
fi

cd "$FOLDER_LOCATION"

# Sort files based on the number between the first two underscores
sorted_files=($(ls *.png | sort -t '_' -k2n))

# Swapping the red and blue channels for each file and adding them to a new array
prepared_files=()
for file in "${sorted_files[@]}"; do
    # Swap the red (R) and blue (B) channels to correct the color space
    corrected_file="corrected_$file"
    convert "$file" -channel RGB -separate -channel BGR -combine "$corrected_file"
    prepared_files+=("$corrected_file")
done

# Calculate delay for approximately 60 FPS
# 100 / 60 = 1.67, ImageMagick's delay unit is in hundredths of a second. Use 2 for simplicity.
convert -delay 2 -loop 0 "${prepared_files[@]}" "$OUTPUT_FILENAME"

# Cleanup: remove the corrected temporary files
rm corrected_*.png

echo "GIF created successfully: $OUTPUT_FILENAME"
