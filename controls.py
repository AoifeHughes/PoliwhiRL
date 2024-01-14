# -*- coding: utf-8 -*-
from pyboy import PyBoy, WindowEvent
import glob
import os
import memory
import OCR
import numpy as np


class Controller:
    def __init__(self, rom_path):
        self.pyboy = PyBoy(rom_path, window_scale=1)
        self.pyboy.set_emulation_speed(target_speed=0)
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

    def handleMovement(self, movement, ticks_per_input=30, wait=60):
        if movement != "PASS":
            self.pyboy.send_input(self.event_dict_press[movement])
            [self.pyboy.tick() for _ in range(ticks_per_input)]
            self.pyboy.send_input(self.event_dict_release[movement])
        else:
            [self.pyboy.tick() for _ in range(ticks_per_input)]

        [self.pyboy.tick() for _ in range(wait)]

    def screen_image(self):
        return self.pyboy.botsupport_manager().screen().screen_image()

    def get_memory_value(self, address):
        return self.pyboy.get_memory_value(address)

    def screen_size(self):
        return self.pyboy.botsupport_manager().screen().screen_ndarray().shape[:2]

    def stop(self, save=True):
        self.pyboy.stop(save)

        # Delete files with .ram extension
        for file in glob.glob("*.ram"):
            os.remove(file)

        # Delete files with .rtc extension
        for file in glob.glob("*.rtc"):
            os.remove(file)

    def get_current_location(self):
        return self.pyboy.get_memory_value(memory.location)

    def has_gotten_out_of_house(self):
        return self.pyboy.get_memory_value(memory.outside_house) == 4

    def player_received(self):
        return self.pyboy.get_memory_value(memory.recived)

    def get_XY(self):
        x_coord = self.pyboy.get_memory_value(memory.X)
        y_coord = self.pyboy.get_memory_value(memory.Y)
        return x_coord, y_coord

    def get_player_money(self):
        money_bytes = [self.pyboy.get_memory_value(memory.money + i) for i in range(3)]
        money = self.bytes_to_int(money_bytes[::-1])
        return money

    def bytes_to_int(self, byte_list):
        return int.from_bytes(byte_list, byteorder="little")

    def party_info(self):
        num_pokemon = self.pyboy.get_memory_value(memory.party_base)
        total_level = 0
        total_hp = 0
        total_exp = 0
        for i in range(num_pokemon):
            base_address = memory.party_base + 0x30 * i
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
