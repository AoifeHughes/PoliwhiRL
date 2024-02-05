# -*- coding: utf-8 -*-
from pyboy import PyBoy
import numpy as np

from PIL import Image
import pytesseract


def preprocess_image(image):
    # Scale the image by a factor of 2
    original_size = image.size
    scaled_size = tuple(2 * x for x in original_size)
    image = image.resize(scaled_size, Image.LANCZOS)
    # Convert to grayscale for better processing
    image = image.convert("L")
    # Apply threshold to get a binary image
    threshold_value = 120  # You might need to adjust this
    image = image.point(lambda p: p > threshold_value and 255)
    # Additional filters can be applied if necessary
    # image = image.filter(ImageFilter.MedianFilter())
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


pyboy = PyBoy("Pokemon - Crystal Version.gbc", window_scale=1)
pyboy.set_emulation_speed(target_speed=0)
unique_locations = set()
unique_XY = set()

while not pyboy.tick():
    money_address = 0xD84E
    money_bytes = [pyboy.get_memory_value(money_address + i) for i in range(3)]
    money = bytes_to_int(money_bytes[::-1])
    print(f"Player's Money: {money} (Raw bytes: {money_bytes})")
    # DCB7 – Y coordinate on overworld map
    # DCB8 – X coordinate on overworld map
    x_coord = pyboy.get_memory_value(0xDCB8)
    y_coord = pyboy.get_memory_value(0xDCB7)
    johto_badges = pyboy.get_memory_value(0xD857)
    kanto_badges = pyboy.get_memory_value(0xD858)
    print(
        f"Johto Badges: {johto_badges}, Kanto Badges: {kanto_badges} (Raw bytes: {[johto_badges, kanto_badges]})"
    )

    num_pokemon = pyboy.get_memory_value(0xDCD7)
    print(f"Number of Party Pokémon: {num_pokemon} (Raw byte: {num_pokemon})")

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
            f"Pokémon {i+1}: Level {level} (Level Raw: {level}, HP Raw: {hp_raw}, EXP Raw: {exp_raw})"
        )
    received = pyboy.get_memory_value(0xCF60)
    print(f"Total Level of Party: {total_level}")
    print(
        f"Player's X Coordinate: {x_coord}, Player's Y Coordinate: {y_coord} Received: {received}"
    )
    print(f"Player's Location: {pyboy.get_memory_value(0xD148)}")
    caught_pokemon = sum(
        count_set_bits(pyboy.get_memory_value(0xDE99 + i)) for i in range(20)
    )
    print(f"Total Number of Caught Pokémon: {caught_pokemon}")
    unique_locations.add(pyboy.get_memory_value(0xD148))
    unique_XY.add((x_coord, y_coord))

    # Print the contents of the screen's text
    screen_image = pyboy.botsupport_manager().screen().screen_image()
    screen_image = preprocess_image(screen_image)
    text = extract_text(screen_image)
    print(f"Text on screen: {text}")

    # pass 60 ticks to let a full second pass
    for i in range(60):
        pyboy.tick()

print(f"Unique Locations: {unique_locations}")
print(f"Unique XY Coordinates: {unique_XY}")

pyboy.stop()