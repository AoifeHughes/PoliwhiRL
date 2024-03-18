# -*- coding: utf-8 -*-
from pyboy import PyBoy
import os
from pynput import keyboard
import numpy as np

# Function to capture the screen and save it with detailed logging information
def capture_and_save_screen(pyboy, log_data):
    screen_image = np.array(pyboy.screen.image)[
            :, :, :3
        ]
    # Generate a filename using log data (e.g., location, coordinates, and steps)
    current_location = pyboy.memory[0xD148]
    x_coord, y_coord = pyboy.memory[0xDCB8], pyboy.memory[0xDCB7]
    steps = len(
        log_data["entries"]
    )  # Using the length of entries as an approximation for steps
    filename = f"capture_{current_location}_x{x_coord}_y{y_coord}_steps{steps}.png"
    # Save the image
    if not os.path.exists("captures"):
        os.makedirs("captures")
    screen_image.save(os.path.join("captures", filename))
    print(f"Screen captured and saved as {filename}")


# Key listener for 'p' press
def on_press(key, pyboy, log_data):
    try:
        if key.char == "s":  # Check if the pressed key is 's'
            capture_and_save_screen(pyboy, log_data)
    except AttributeError:
        pass  # Do nothing if other keys are pressed


def start_key_listener(pyboy, log_data):
    # Start listening for 'p' key press in the background
    listener = keyboard.Listener(
        on_press=lambda event: on_press(event, pyboy, log_data)
    )
    listener.start()


# Main function (assuming your script starts here)
def main():
    pyboy = PyBoy("emu_files/Pokemon - Crystal Version.gbc", window="SDL2")
    pyboy.set_emulation_speed(target_speed=2)
    log_data = {"entries": []}  # Initialize log data

    start_key_listener(pyboy, log_data)  # Start the key listener

    with open("emu_files/states/start.state", "rb") as state:
        pyboy.load_state(state)

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()
