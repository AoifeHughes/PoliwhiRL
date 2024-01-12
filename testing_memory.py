from pyboy import PyBoy, WindowEvent
import numpy as np
def read_little_endian(pyboy, start, end):
    raw_bytes = []
    for i in range(end, start - 1, -1):
        byte = pyboy.get_memory_value(i)
        raw_bytes.append(byte)
    return raw_bytes

def bytes_to_int(byte_list):
    return int.from_bytes(byte_list, byteorder='little')

def count_set_bits(byte):
    count = 0
    while byte:
        count += byte & 1
        byte >>= 1
    return count

pyboy = PyBoy('Pokemon - Crystal Version.gbc', window_scale=1)
pyboy.set_emulation_speed(target_speed=0)

while not pyboy.tick():
    money_address = 0xD84E
    money_bytes = [pyboy.get_memory_value(money_address + i) for i in range(3)]
    money = bytes_to_int(money_bytes[::-1])
    print(f"Player's Money: {money} (Raw bytes: {money_bytes})")

    johto_badges = pyboy.get_memory_value(0xD857)
    kanto_badges = pyboy.get_memory_value(0xD858)
    print(f"Johto Badges: {johto_badges}, Kanto Badges: {kanto_badges} (Raw bytes: {[johto_badges, kanto_badges]})")

    num_pokemon = pyboy.get_memory_value(0xDCD7)
    print(f"Number of Party Pokémon: {num_pokemon} (Raw byte: {num_pokemon})")

    total_level = 0
    for i in range(num_pokemon):
        base_address = 0xDCDF + 0x30 * i
        level = pyboy.get_memory_value(base_address + 0x1F)
        hp_raw = np.sum(read_little_endian(pyboy, base_address + 0x22, base_address + 0x23) * np.array([1, 256])  )
        exp_raw = np.sum(read_little_endian(pyboy, base_address + 0x08, base_address + 0x0A) * np.array([1, 256, 65536]))
        total_level += level
        print(f"Pokémon {i+1}: Level {level} (Level Raw: {level}, HP Raw: {hp_raw}, EXP Raw: {exp_raw})")

    print(f"Total Level of Party: {total_level}")

    caught_pokemon = sum(count_set_bits(pyboy.get_memory_value(0xDE99 + i)) for i in range(20))
    print(f"Total Number of Caught Pokémon: {caught_pokemon}")

pyboy.stop()
