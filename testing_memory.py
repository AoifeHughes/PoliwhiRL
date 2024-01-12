from pyboy import PyBoy, WindowEvent

def read_little_endian(pyboy, start, end):
    value = 0
    raw_bytes = []
    for i in range(end, start - 1, -1):
        byte = pyboy.get_memory_value(i)
        raw_bytes.append(byte)
        value = (value << 8) + byte
    return value, raw_bytes

def bcd_to_int(bcd_bytes):
    total = 0
    for byte in bcd_bytes:
        total = (total * 100) + (byte // 16 * 10) + (byte % 16)
    return total

def count_set_bits(byte):
    count = 0
    while byte:
        count += byte & 1
        byte >>= 1
    return count

pyboy = PyBoy('Pokemon - Crystal Version.gbc', window_scale=1)
pyboy.set_emulation_speed(0)

while not pyboy.tick():
    money_raw = [pyboy.get_memory_value(i) for i in range(0xD84E, 0xD851)]
    money = bcd_to_int(money_raw)
    print(f"Player's Money: {money} (Raw bytes: {money_raw})")

    johto_badges = pyboy.get_memory_value(0xD857)
    kanto_badges = pyboy.get_memory_value(0xD858)
    print(f"Johto Badges: {johto_badges}, Kanto Badges: {kanto_badges} (Raw bytes: {[johto_badges, kanto_badges]})")

    num_pokemon = pyboy.get_memory_value(0xDCD7)
    print(f"Number of Party Pokémon: {num_pokemon} (Raw byte: {num_pokemon})")

    total_level = 0
    for i in range(num_pokemon):
        base_address = 0xD16C + 0x30 * i
        level = pyboy.get_memory_value(base_address + 0x1F)
        hp, hp_raw = read_little_endian(pyboy, base_address + 0x21, base_address + 0x22)
        total_level += level
        print(f"Pokémon {i+1}: Level {level}, HP {hp} (Level Raw: {level}, HP Raw: {hp_raw})")

    print(f"Total Level of Party: {total_level}")

    caught_pokemon = sum(count_set_bits(pyboy.get_memory_value(0xDE99 + i)) for i in range(20))
    print(f"Total Number of Caught Pokémon: {caught_pokemon}")

pyboy.stop()
