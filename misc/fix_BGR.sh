convert_bgr_to_rgb() {
    # Get the directory path from the function argument
    local root_directory="$1"

    # Find all PNG files in the specified directory and its subdirectories
    find "$root_directory" -type f -name "*.png" | while read -r file; do
        # Use ImageMagick's convert command to swap the red and blue channels
        # This effectively converts an image from BGR to RGB
        # The converted image overwrites the original file
        convert "$file" -channel RGB -separate -channel BGR -combine "$file"
    done

    echo "All PNG files have been converted from BGR to RGB."
}

convert_bgr_to_rgb "runs/DoubleDQN/2"
