# -*- coding: utf-8 -*-
import io
from pyboy import PyBoy, WindowEvent
import os
import PoliwhiRL.environment.RAM_locations as RAM_locations
import PoliwhiRL.utils.OCR as OCR
import numpy as np
import shutil
import tempfile


class Controller:
    def __init__(self, rom_path, state_path=None):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        rom_root = os.path.dirname(rom_path)
        # copy other files to the temporary directory
        paths = [
            shutil.copy(file, self.temp_dir)
            for file in [
                rom_path,
                state_path,
                f"{rom_root}/Pokemon - Crystal Version.gbc.ram",
                f"{rom_root}/Pokemon - Crystal Version.gbc.rtc",
            ]
            if file is not None
        ]

        # Initialize PyBoy with the ROM in the temporary directory
        self.pyboy = PyBoy(paths[0], debug=False, window_type="headless")
        self.pyboy.set_emulation_speed(0)
        if state_path is not None:
            with open(paths[1], "rb") as stateFile:
                self.pyboy.load_state(stateFile)

        self.movements = [
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "A",
            "B",
            "START",
            "SELECT",
            "PASS",
        ]

        self.event_dict_press = {
            "UP": WindowEvent.PRESS_ARROW_UP,
            "DOWN": WindowEvent.PRESS_ARROW_DOWN,
            "LEFT": WindowEvent.PRESS_ARROW_LEFT,
            "RIGHT": WindowEvent.PRESS_ARROW_RIGHT,
            "A": WindowEvent.PRESS_BUTTON_A,
            "B": WindowEvent.PRESS_BUTTON_B,
            "START": WindowEvent.PRESS_BUTTON_START,
            "SELECT": WindowEvent.PRESS_BUTTON_SELECT,
        }

        self.event_dict_release = {
            "UP": WindowEvent.RELEASE_ARROW_UP,
            "DOWN": WindowEvent.RELEASE_ARROW_DOWN,
            "LEFT": WindowEvent.RELEASE_ARROW_LEFT,
            "RIGHT": WindowEvent.RELEASE_ARROW_RIGHT,
            "A": WindowEvent.RELEASE_BUTTON_A,
            "B": WindowEvent.RELEASE_BUTTON_B,
            "START": WindowEvent.RELEASE_BUTTON_START,
            "SELECT": WindowEvent.RELEASE_BUTTON_SELECT,
        }

    def save_state(self, file):
        self.pyboy.save_state(file)

    def handleMovement(self, movement, ticks_per_input=10, wait=480):
        self.pyboy._rendering(False)
        if movement != "PASS":
            self.pyboy.send_input(self.event_dict_press[movement])
            [self.pyboy.tick() for _ in range(ticks_per_input)]
            self.pyboy.send_input(self.event_dict_release[movement])
        else:
            [self.pyboy.tick() for _ in range(ticks_per_input)]
        [self.pyboy.tick() for _ in range(wait)]
        self.pyboy._rendering(True)
        self.pyboy.tick()

    def screen_image(self):
        return self.pyboy.botsupport_manager().screen().screen_image()

    def get_memory_value(self, address):
        return self.pyboy.get_memory_value(address)

    def screen_size(self):
        return self.pyboy.botsupport_manager().screen().screen_ndarray().shape[:2]

    def stop(self, save=True):
        self.pyboy.stop(save)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def get_current_location(self):
        return self.pyboy.get_memory_value(RAM_locations.location)

    def has_gotten_out_of_house(self):
        return self.pyboy.get_memory_value(RAM_locations.outside_house) == 4

    def player_received(self):
        return self.pyboy.get_memory_value(RAM_locations.received)

    def get_XY(self):
        x_coord = self.pyboy.get_memory_value(RAM_locations.X)
        y_coord = self.pyboy.get_memory_value(RAM_locations.Y)
        return x_coord, y_coord

    def get_player_money(self):
        money_bytes = [
            self.pyboy.get_memory_value(RAM_locations.money + i) for i in range(3)
        ]
        money = self.bytes_to_int(money_bytes[::-1])
        return money

    def bytes_to_int(self, byte_list):
        return int.from_bytes(byte_list, byteorder="little")

    def read_little_endian(self, start, end):
        raw_bytes = []
        for i in range(end, start - 1, -1):
            byte = self.pyboy.get_memory_value(i)
            raw_bytes.append(byte)
        return raw_bytes

    def party_info(self):
        num_pokemon = self.pyboy.get_memory_value(RAM_locations.num_pokemon)
        total_level = 0
        total_hp = 0
        total_exp = 0
        for i in range(num_pokemon):
            base_address = RAM_locations.party_base + 0x30 * i
            level = self.pyboy.get_memory_value(base_address + 0x1F)
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
        return total_level, total_hp, total_exp

    def get_text_on_screen(self):
        screen_image = self.pyboy.botsupport_manager().screen().screen_image()
        text = OCR.extract_text(OCR.preprocess_image(screen_image))
        return text

    def create_memory_state(self, controller):
        virtual_file = io.BytesIO()
        self.save_state(virtual_file)
        return virtual_file

    def store_state(self, state, i):
        # check states folder exists and create if not
        state.seek(0)
        if not os.path.isdir("./states"):
            os.mkdir("./states")
        with open(f"./states/state_{i}.state", "wb") as f:
            f.write(state.read())
