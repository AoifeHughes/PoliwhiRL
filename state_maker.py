import pyboy
import time

# Initialize PyBoy with a ROM
pyboy = pyboy.PyBoy('Pokemon - Crystal Version.gbc')

# Make sure to set PyBoy to headless if you don't need a GUI
pyboy.set_emulation_speed(0)  # 0 for unbounded speed, 1 for normal speed

# Main loop
try:
    while not pyboy.tick():
        current_time = time.time()
        # Check if two minutes have passed
        if current_time % 120 < 1:
            # Save the state to a file
            pyboy.save_state('saved_state.state')
            # Wait a little to avoid saving the state multiple times in one minute
            time.sleep(1)
            break
finally:
    pyboy.stop()
