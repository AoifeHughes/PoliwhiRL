# -*- coding: utf-8 -*-
from pyboy import PyBoy
import numpy as np
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import os
import json
from pynput import keyboard

# Function to capture the screen and save it with detailed logging information
def capture_and_save_screen(pyboy, log_data):
    screen_image = pyboy.botsupport_manager().screen().screen_image()
    # Generate a filename using log data (e.g., location, coordinates, and steps)
    current_location = pyboy.get_memory_value(0xD148)
    x_coord, y_coord = pyboy.get_memory_value(0xDCB8), pyboy.get_memory_value(0xDCB7)
    steps = len(log_data["entries"])  # Using the length of entries as an approximation for steps
    filename = f"capture_{current_location}_x{x_coord}_y{y_coord}_steps{steps}.png"
    # Save the image
    if not os.path.exists("captures"):
        os.makedirs("captures")
    screen_image.save(os.path.join("captures", filename))
    print(f"Screen captured and saved as {filename}")

# Key listener for 'p' press
def on_press(key, pyboy, log_data):
    try:
        if key.char == 's':  # Check if the pressed key is 's'
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
    pyboy = PyBoy("emu_files/Pokemon - Crystal Version.gbc", window_scale=1)
    pyboy.set_emulation_speed(target_speed=2)
    log_data = {"entries": []}  # Initialize log data

    start_key_listener(pyboy, log_data)  # Start the key listener

    with open("emu_files/states/start.state", "rb") as state:
        pyboy.load_state(state)

    while not pyboy.tick():
        # Your game logic here
        pass

    pyboy.stop()

if __name__ == "__main__":
    main()
