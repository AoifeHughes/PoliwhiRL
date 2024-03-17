import numpy as np

# -*- coding: utf-8 -*-


class RAMManagement:
    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.location = 0xD148
        self.X = 0xDCB8
        self.Y = 0xDCB7
        self.stairs = (7, 1)
        self.frontdoor1 = (7, 7)
        self.frontdoor2 = (6, 7)
        self.received = 0xCF60
        self.money = 0xD84E
        self.party_base = 0xDCDF
        self.num_pokemon = 0xDCD7

        self.locations = {
            "DownstairsPlayersHouse": 6,
            "OutsideStartingArea": 4,
            "ProfessorElmsLab": 5,
            "NPC_house1": 8,
            "NPC_house2": 9,
            "MrPokemonsHouse": 10,
        }

        self.pokedex_seen = (0xDEB9, 0xDED8)
        self.pokedex_owned = (0xDE99, 0xDEB8)

    def get_memory_value(self, address):
        return self.pyboy.memory[address]

    def get_current_location(self):
        loc = self.get_memory_value(self.location)
        if loc == 7:  # starting zone is also 7 if you leave and come back
            loc = 0
        return loc

    def has_gotten_out_of_house(self):
        return self.get_memory_value(self.outside_house) == 4

    def player_received(self):
        return self.get_memory_value(self.received)

    def get_XY(self):
        x_coord = self.get_memory_value(self.X)
        y_coord = self.get_memory_value(self.Y)
        return x_coord, y_coord

    def get_player_money(self):
        money_bytes = [
            self.get_memory_value(self.money + i) for i in range(3)
        ]
        money = self.bytes_to_int(money_bytes[::-1])
        return money

    def bytes_to_int(self, byte_list):
        return int.from_bytes(byte_list, byteorder="little")

    def read_little_endian(self, start, end):
        raw_bytes = []
        for i in range(end, start - 1, -1):
            byte = self.get_memory_value(i)
            raw_bytes.append(byte)
        return raw_bytes

    def get_party_info(self):
        num_pokemon = self.get_memory_value(self.num_pokemon)
        total_level = 0
        total_hp = 0
        total_exp = 0
        for i in range(num_pokemon):
            base_address = self.party_base + 0x30 * i
            level = self.get_memory_value(base_address + 0x1F)
            hp = np.sum(
                self.read_little_endian(base_address + 0x22, base_address + 0x23)
                * np.array([1, 256])
            )
            exp = np.sum(
                self.read_little_endian(base_address + 0x08, base_address + 0x0A)
                * np.array([1, 256, 65536])
            )
            total_level += level
            total_hp += hp
            total_exp += exp
        return int(total_level), int(total_hp), int(total_exp)

    def get_pkdex_seen(self):
        start_address, end_address = self.pokedex_seen

        total_seen = 0
        for address in range(start_address, end_address + 1):
            # Retrieve the byte value from the current address
            byte_value = self.get_memory_value(address)

            # Count the number of bits set to 1 (i.e., Pokémon seen) in this byte
            while byte_value:
                total_seen += byte_value & 1
                byte_value >>= 1  # Right shift to process the next bit

        return total_seen

    def get_pkdex_owned(self):
        start_address, end_address = self.pokedex_owned

        total_owned = 0
        for address in range(start_address, end_address + 1):
            # Retrieve the byte value from the current address
            byte_value = self.get_memory_value(address)

            # Count the number of bits set to 1 (i.e., Pokémon owned) in this byte
            while byte_value:
                total_owned += byte_value & 1
                byte_value >>= 1
        return total_owned
    
    def update_variables(self):
        self.money = self.get_player_money()
        self.location = self.get_current_location()
        self.X, self.Y = self.get_XY()
        self.party_info = self.get_party_info()
        self.pkdex_seen = self.get_pkdex_seen()
        self.pkdex_owned = self.get_pkdex_owned()

    def get_variables(self):
        # return a dict of all the variables
        self.update_variables()
        return {
            "money": self.money,
            "location": self.location,
            "X": self.X,
            "Y": self.Y,
            "party_info": self.party_info,
            "pkdex_seen": self.pkdex_seen,
            "pkdex_owned": self.pkdex_owned,
        }