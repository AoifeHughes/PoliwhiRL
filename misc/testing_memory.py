# -*- coding: utf-8 -*-
from pyboy import PyBoy
import numpy as np
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import os
import json


def log_data_to_json(log_data, log_file_path="log_data.json"):
    with open(log_file_path, "w") as log_file:
        json.dump(log_data, log_file, indent=4)


# Initialize a log structure
log_data = {"entries": []}
log_id = 0  # Starting ID for your log entries


def preprocess_image(image):
    original_size = image.size
    scaled_size = tuple(2 * x for x in original_size)
    image = image.resize(scaled_size, Image.LANCZOS)
    image = image.convert("L")
    threshold_value = 120
    image = image.point(lambda p: p > threshold_value and 255)
    return image


def extract_text(image):
    text = pytesseract.image_to_string(image, config="--psm 11")
    return text


def read_little_endian(pyboy, start, end):
    raw_bytes = []
    for i in range(end, start - 1, -1):
        byte = pyboy.get_memory_value(i)
        raw_bytes.append(byte)
    return raw_bytes


def bytes_to_int(byte_list):
    return int.from_bytes(byte_list, byteorder="little")


def count_set_bits(byte):
    count = 0
    while byte:
        count += byte & 1
        byte >>= 1
    return count


def plot_coordinates(unique_XY_by_location):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_XY_by_location)))
    for (location, coordinates), color in zip(unique_XY_by_location.items(), colors):
        X = [coord[0] for coord in coordinates]
        Y = [coord[1] for coord in coordinates]
        fig, ax = plt.subplots(1, figsize=(5, 5), dpi=100)
        ax.scatter(X, Y, alpha=0.6, edgecolors="w", color=color)
        # Only add the label to the line plot
        ax.plot(X, Y, alpha=0.5, color=color, label=f"Location {location}")
        fig.savefig(f"testing_files/{location}/location_{location}.png")
        fig.suptitle("Ordered XY Coordinates by Location")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        # Ensure the legend is displayed. The 'best' location will automatically adjust to a less obtrusive place.
        fig.legend()


pyboy = PyBoy("emu_files/Pokemon - Crystal Version.gbc", window_scale=1)
pyboy.set_emulation_speed(target_speed=2)
unique_XY_by_location = {}
imgs = {}

with open("emu_files/states/start.state", "rb") as state:
    pyboy.load_state(state)

while not pyboy.tick():
    money_address = 0xD84E
    money_bytes = [pyboy.get_memory_value(money_address + i) for i in range(3)]
    money = bytes_to_int(money_bytes[::-1])
    print(f"Player's Money: {money} (Raw bytes: {money_bytes})\n")

    x_coord = pyboy.get_memory_value(0xDCB8)
    y_coord = pyboy.get_memory_value(0xDCB7)
    johto_badges = pyboy.get_memory_value(0xD857)
    kanto_badges = pyboy.get_memory_value(0xD858)
    print(
        f"Johto Badges: {johto_badges}, Kanto Badges: {kanto_badges} (Raw bytes: {[johto_badges, kanto_badges]})\n"
    )

    num_pokemon = pyboy.get_memory_value(0xDCD7)
    print(f"Number of Party Pokémon: {num_pokemon} (Raw byte: {num_pokemon})\n")

    total_level = 0
    for i in range(num_pokemon):
        base_address = 0xDCDF + 0x30 * i
        level = pyboy.get_memory_value(base_address + 0x1F)
        hp_raw = np.sum(
            read_little_endian(pyboy, base_address + 0x22, base_address + 0x23)
            * np.array([1, 256])
        )
        exp_raw = np.sum(
            read_little_endian(pyboy, base_address + 0x08, base_address + 0x0A)
            * np.array([1, 256, 65536])
        )
        total_level += level
        print(
            f"Pokémon {i+1}: Level {level} (Level Raw: {level}, HP Raw: {hp_raw}, EXP Raw: {exp_raw})\n"
        )

    received = pyboy.get_memory_value(0xCF60)
    print(f"Total Level of Party: {total_level}\n")
    print(
        f"Player's X Coordinate: {x_coord}, Player's Y Coordinate: {y_coord} Received: {received}\n"
    )
    print(f"Player's Location: {pyboy.get_memory_value(0xD148)}\n")
    caught_pokemon = sum(
        count_set_bits(pyboy.get_memory_value(0xDE99 + i)) for i in range(20)
    )
    print(f"Total Number of Caught Pokémon: {caught_pokemon}\n")
    current_location = pyboy.get_memory_value(0xD148)

    if current_location not in unique_XY_by_location:
        unique_XY_by_location[current_location] = []

    if current_location not in imgs:
        imgs[current_location] = []
    unique_XY_by_location[current_location].append((x_coord, y_coord))
    imgs[current_location].append(pyboy.botsupport_manager().screen().screen_image())


if not os.path.exists("testing_files"):
    os.makedirs("testing_files")

for loc, img_list in imgs.items():
    if not os.path.exists(f"testing_files/{loc}"):
        os.makedirs(f"testing_files/{loc}")
    for i, img in enumerate(img_list):
        # Fetch the corresponding coordinates for the current image
        x_coord, y_coord = unique_XY_by_location[loc][i]

        # Update the img_path to include location, X, and Y coordinates in the filename
        img_path = f"testing_files/{loc}/{loc}_x{x_coord}_y{y_coord}_{i}.png"
        img.save(img_path)

        # Prepare log entry with coordinates, steps, location, and the image path
        entry = {
            "id": log_id,
            "location": loc,
            "coordinates": (
                x_coord,
                y_coord,
            ),  # Fetch the coordinates directly for clarity
            "steps": i,  # Assuming each image represents a step
            "image_path": img_path,
        }
        log_data["entries"].append(entry)

        # Increment log ID for next entry
        log_id += 1
# After collecting all data, write the log data to a JSON file
log_data_to_json(log_data)

plot_coordinates(unique_XY_by_location)

pyboy.stop()
