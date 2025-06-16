# -*- coding: utf-8 -*-
import numpy as np


class RAMManagement:
    def __init__(self, pyboy):
        self.pyboy = pyboy

        # Memory locations
        self.room_player_is_in = 0xD148
        self.map_number = 0xDCB6
        self.overworld_X = 0xDCB8
        self.overworld_Y = 0xDCB7
        self.received = 0xCF60
        self.player_money = 0xD84E
        self.party_base = 0xDCDF
        self.num_pokemon_in_party = 0xDCD7
        self.pokedex_seen = (0xDEB9, 0xDED8)
        self.pokedex_owned = (0xDE99, 0xDEB8)
        self.screen_tile_start = 0xC4A0
        self.screen_tile_end = 0xC607

        self.warp_number = 0xDCB4
        self.map_bank = 0xDCB5

        self.wram_start = 0xC000
        self.wram_end = 0xDFFF
        self.wram_size = self.wram_end - self.wram_start + 1

        # collision data
        self.collision_down = 0xC2FA
        self.collision_up = 0xC2FB
        self.collision_left = 0xC2FC
        self.collision_right = 0xC2FD

        # Additional player state
        self.player_name_start = 0xD47D
        self.player_name_end = 0xD486
        self.trainer_id = 0xD47B
        self.play_time_start = 0xD4C4
        self.play_time_end = 0xD4C8
        
        # Badge progression
        self.johto_badges = 0xD857
        self.kanto_badges = 0xD858
        
        # Battle state
        self.battle_type = 0xD22D
        self.battle_turn_counter = 0xCCD5
        self.enemy_pokemon_species = 0xD204
        self.enemy_pokemon_level = 0xD213
        self.enemy_held_item = 0xD207
        self.enemy_moves_start = 0xD208
        self.enemy_moves_end = 0xD20B
        self.enemy_hp_current = 0xD216
        self.enemy_hp_max = 0xD218
        self.enemy_stats_start = 0xD21A  # Attack, Defense, Speed, SpAtk, SpDef
        self.player_stat_modifiers_start = 0xCD1A
        self.enemy_stat_modifiers_start = 0xCD2E
        
        # Wild encounter info
        self.wild_pokemon_species = 0xD0ED
        self.wild_pokemon_level = 0xD0FC
        
        # Items and inventory
        self.bag_start = 0xD89F
        self.repel_steps = 0xDCA1
        self.casino_coins = 0xD855
        
        # Menu states
        self.current_menu_item = 0xCC26
        self.last_menu_item = 0xCC28
        self.battle_menu_state = 0xCC2D
        
        # RNG system
        self.rng_add = 0xFFE3
        self.rng_sub = 0xFFE4
        
        # Movement
        self.joypad_override = 0xCD38

    def get_memory_value(self, address):
        return self.pyboy.memory[address]

    def set_memory_value(self, address, value):
        self.pyboy.memory[address] = value

    def get_XY(self):
        x_coord = self.get_memory_value(self.overworld_X)
        y_coord = self.get_memory_value(self.overworld_Y)
        return x_coord, y_coord

    def get_player_money(self):
        money_bytes = [self.get_memory_value(self.player_money + i) for i in range(3)]
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
        num_pokemon = self.get_memory_value(self.num_pokemon_in_party)
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

    def get_party_pokemon_details(self, pokemon_index):
        """Get detailed info for a specific Pokemon in the party"""
        if pokemon_index >= self.get_memory_value(self.num_pokemon_in_party):
            return None
            
        base_address = self.party_base + 0x30 * pokemon_index
        
        return {
            "species": self.get_memory_value(base_address),
            "held_item": self.get_memory_value(base_address + 0x01),
            "moves": [self.get_memory_value(base_address + 0x02 + i) for i in range(4)],
            "level": self.get_memory_value(base_address + 0x1F),
            "hp_current": self.get_memory_value(base_address + 0x22) + (self.get_memory_value(base_address + 0x23) << 8),
            "hp_max": self.get_memory_value(base_address + 0x24) + (self.get_memory_value(base_address + 0x25) << 8),
            "attack": self.get_memory_value(base_address + 0x26) + (self.get_memory_value(base_address + 0x27) << 8),
            "defense": self.get_memory_value(base_address + 0x28) + (self.get_memory_value(base_address + 0x29) << 8),
            "speed": self.get_memory_value(base_address + 0x2A) + (self.get_memory_value(base_address + 0x2B) << 8),
            "special_attack": self.get_memory_value(base_address + 0x2C) + (self.get_memory_value(base_address + 0x2D) << 8),
            "special_defense": self.get_memory_value(base_address + 0x2E) + (self.get_memory_value(base_address + 0x2F) << 8),
            "pp": [self.get_memory_value(base_address + 0x17 + i) for i in range(4)]
        }

    def get_pokedex_seen(self):
        start_address, end_address = self.pokedex_seen
        total_seen = 0
        for address in range(start_address, end_address + 1):
            byte_value = self.get_memory_value(address)
            while byte_value:
                total_seen += byte_value & 1
                byte_value >>= 1
        return total_seen

    def get_pokedex_owned(self):
        start_address, end_address = self.pokedex_owned
        total_owned = 0
        for address in range(start_address, end_address + 1):
            byte_value = self.get_memory_value(address)
            while byte_value:
                total_owned += byte_value & 1
                byte_value >>= 1
        return total_owned

    def get_map_num(self):
        return self.get_memory_value(self.map_number)

    def get_badges(self):
        """Get badge information as dictionaries with badge names"""
        johto = self.get_memory_value(self.johto_badges)
        kanto = self.get_memory_value(self.kanto_badges)
        
        johto_badges = {
            "falkner": bool(johto & 0x01),
            "bugsy": bool(johto & 0x02),
            "whitney": bool(johto & 0x04),
            "morty": bool(johto & 0x08),
            "jasmine": bool(johto & 0x10),
            "chuck": bool(johto & 0x20),
            "pryce": bool(johto & 0x40),
            "clair": bool(johto & 0x80)
        }
        
        kanto_badges = {
            "brock": bool(kanto & 0x01),
            "misty": bool(kanto & 0x02),
            "surge": bool(kanto & 0x04),
            "erika": bool(kanto & 0x08),
            "janine": bool(kanto & 0x10),
            "sabrina": bool(kanto & 0x20),
            "blaine": bool(kanto & 0x40),
            "blue": bool(kanto & 0x80)
        }
        
        return johto_badges, kanto_badges

    def get_badge_count(self):
        """Get total number of badges earned"""
        johto = bin(self.get_memory_value(self.johto_badges)).count('1')
        kanto = bin(self.get_memory_value(self.kanto_badges)).count('1')
        return johto + kanto

    def get_battle_state(self):
        """Get current battle information if in battle"""
        battle_type = self.get_memory_value(self.battle_type)
        if battle_type == 0:  # Not in battle
            return None
            
        return {
            "type": battle_type,
            "turn": self.get_memory_value(self.battle_turn_counter),
            "enemy_species": self.get_memory_value(self.enemy_pokemon_species),
            "enemy_level": self.get_memory_value(self.enemy_pokemon_level),
            "enemy_held_item": self.get_memory_value(self.enemy_held_item),
            "enemy_moves": [self.get_memory_value(self.enemy_moves_start + i) for i in range(4)],
            "enemy_hp_current": self.get_memory_value(self.enemy_hp_current) + (self.get_memory_value(self.enemy_hp_current + 1) << 8),
            "enemy_hp_max": self.get_memory_value(self.enemy_hp_max) + (self.get_memory_value(self.enemy_hp_max + 1) << 8),
            "enemy_stats": {
                "attack": self.get_memory_value(self.enemy_stats_start) + (self.get_memory_value(self.enemy_stats_start + 1) << 8),
                "defense": self.get_memory_value(self.enemy_stats_start + 2) + (self.get_memory_value(self.enemy_stats_start + 3) << 8),
                "speed": self.get_memory_value(self.enemy_stats_start + 4) + (self.get_memory_value(self.enemy_stats_start + 5) << 8),
                "special_attack": self.get_memory_value(self.enemy_stats_start + 6) + (self.get_memory_value(self.enemy_stats_start + 7) << 8),
                "special_defense": self.get_memory_value(self.enemy_stats_start + 8) + (self.get_memory_value(self.enemy_stats_start + 9) << 8)
            }
        }

    def get_player_name(self):
        """Get the player's name"""
        name_bytes = []
        for i in range(10):  # Max name length
            byte = self.get_memory_value(self.player_name_start + i)
            if byte == 0x50:  # Terminator
                break
            name_bytes.append(byte)
        # Convert from Pokemon character encoding to ASCII (simplified)
        # Note: Proper conversion would need a full character map
        return bytes(name_bytes)

    def get_trainer_id(self):
        """Get the trainer ID"""
        return self.get_memory_value(self.trainer_id) + (self.get_memory_value(self.trainer_id + 1) << 8)

    def get_repel_steps(self):
        """Get remaining repel steps"""
        return self.get_memory_value(self.repel_steps)

    def get_casino_coins(self):
        """Get casino coins"""
        return self.get_memory_value(self.casino_coins) + (self.get_memory_value(self.casino_coins + 1) << 8)

    def get_rng_state(self):
        """Get current RNG state for prediction/manipulation"""
        return {
            "add": self.get_memory_value(self.rng_add),
            "sub": self.get_memory_value(self.rng_sub)
        }

    def export_wram(self):
        """
        Export the entire Work RAM (WRAM) as a numpy array.
        This includes both WRAM Bank 0 (C000-CFFF) and WRAM Bank 1 (D000-DFFF).
        """
        wram_data = np.zeros(self.wram_size, dtype=np.uint8)

        for i in range(self.wram_size):
            wram_data[i] = self.get_memory_value(self.wram_start + i)

        return wram_data

    def get_screen_tiles(self):
        # This is basically the static background ...

        # The screen is 20 tiles wide and 18 tiles high
        screen_width = 20
        screen_height = 18

        # Create a 2D numpy array to store the tile values
        screen_tiles = np.zeros((screen_height, screen_width), dtype=np.uint8)

        # Read the tile values from memory and populate the array
        for i in range(screen_height):
            for j in range(screen_width):
                mem_address = self.screen_tile_start + i * screen_width + j
                screen_tiles[i, j] = self.get_memory_value(mem_address)

        return screen_tiles

    def get_variables(self):
        x, y = self.get_XY()
        johto_badges, kanto_badges = self.get_badges()
        
        variables = {
            "money": self.get_player_money(),
            "X": x,
            "Y": y,
            "party_info": self.get_party_info(),
            "pokedex_seen": self.get_pokedex_seen(),
            "pokedex_owned": self.get_pokedex_owned(),
            "map_num": self.get_map_num(),
            "warp_number": self.get_memory_value(self.warp_number),
            "map_bank": self.get_memory_value(self.map_bank),
            "room": self.get_memory_value(self.room_player_is_in),
            "collision_down": self.get_memory_value(self.collision_down),
            "collision_up": self.get_memory_value(self.collision_up),
            "collision_left": self.get_memory_value(self.collision_left),
            "collision_right": self.get_memory_value(self.collision_right),
            # New additions
            "badge_count": self.get_badge_count(),
            "johto_badges": johto_badges,
            "kanto_badges": kanto_badges,
            "trainer_id": self.get_trainer_id(),
            "repel_steps": self.get_repel_steps(),
            "casino_coins": self.get_casino_coins(),
            "battle_state": self.get_battle_state(),
            "rng_state": self.get_rng_state(),
            "joypad_override": self.get_memory_value(self.joypad_override)
        }
        
        # Add party pokemon details if needed
        num_pokemon = self.get_memory_value(self.num_pokemon_in_party)
        if num_pokemon > 0:
            variables["lead_pokemon"] = self.get_party_pokemon_details(0)
            
        return variables