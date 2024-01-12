from pyboy import PyBoy

def bytes_to_int(byte_list):
    return int.from_bytes(byte_list, byteorder='little')

def read_pokemon_data(pyboy, base_address):
    current_hp_bytes = [
        pyboy.get_memory_value(base_address),
        pyboy.get_memory_value(base_address + 1)
    ]
    max_hp_bytes = [
        pyboy.get_memory_value(base_address + 2),
        pyboy.get_memory_value(base_address + 3)
    ]
    
    current_hp = bytes_to_int(current_hp_bytes)
    max_hp = bytes_to_int(max_hp_bytes)
    level = pyboy.get_memory_value(base_address + 30)  # Level is at an offset of 30 bytes

    return {
        'Current HP': current_hp,
        'Max HP': max_hp,
        'Level': level
    }


# Initialize PyBoy and load your ROM
pyboy = PyBoy('Pokemon - Crystal Version.gbc', window_scale=1)
pyboy.set_emulation_speed(0)

# Addresses
money_address = 0xD84E
badges_johto_address = 0xD857
badges_kanto_address = 0xD858
party_size_address = 0xDCD7
party_data_base_address = 0xDCDF
pokemon_data_size = 44  # The size of the data block for each Pokemon

previous_data = {}

while not pyboy.tick():
    # Reading Money
    money_bytes = [pyboy.get_memory_value(money_address + i) for i in range(3)]
    money = bytes_to_int(money_bytes[::-1])

    # Reading Badges
    badges_johto = pyboy.get_memory_value(badges_johto_address)
    badges_kanto = pyboy.get_memory_value(badges_kanto_address)

    # Reading Party Size and Data
    party_size = pyboy.get_memory_value(party_size_address)
    party_data = {}
    for i in range(party_size):
        pokemon_data_address = party_data_base_address + i * pokemon_data_size
        party_data[f'Pokemon {i + 1}'] = read_pokemon_data(pyboy, pokemon_data_address)

    # Creating a combined dictionary
    current_data = {
        'Money': money,
        'Johto Badges': badges_johto,
        'Kanto Badges': badges_kanto,
        'Party': party_data
    }

    # Check for changes
    if current_data != previous_data:
        print(current_data)
        previous_data = current_data.copy()

    # Uncomment the line below if you want the script to run continuously
    # Otherwise, it will run only once
    # break

pyboy.stop()
