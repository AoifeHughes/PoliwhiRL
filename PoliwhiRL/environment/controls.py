# -*- coding: utf-8 -*-
import io
from pyboy import PyBoy, WindowEvent
import os
import PoliwhiRL.environment.RAM_locations as RAM_locations
import PoliwhiRL.utils.OCR as OCR
import numpy as np
import shutil
import tempfile
from PoliwhiRL.environment.rewards import calc_rewards
from PoliwhiRL.utils.utils import document
import time
import json
from PoliwhiRL.environment.ImageMemory import ImageMemory


class Controller:
    def __init__(
        self,
        rom_path,
        state_path=None,
        timeout=100,
        log_path="./logs/log.json",
        use_sight=False,
    ):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        rom_root = os.path.dirname(rom_path)
        self.log_path = log_path
        self.state_path = state_path
        self.ogTimeout = timeout
        self.timeout = timeout
        self.timeoutcap = timeout * 100
        self.frames_per_loc = {i: 0 for i in range(256)}
        self.use_sight = use_sight
        # copy other files to the temporary directory
        self.paths = [
            shutil.copy(file, self.temp_dir)
            for file in [
                rom_path,
                self.state_path,
                f"{rom_root}/Pokemon - Crystal Version.gbc.ram",
                f"{rom_root}/Pokemon - Crystal Version.gbc.rtc",
            ]
            if file is not None
        ]

        # Initialize PyBoy with the ROM in the temporary directory
        self.pyboy = PyBoy(self.paths[0], debug=False, window_type="headless")
        self.pyboy.set_emulation_speed(0)
        if self.state_path is not None:
            with open(self.state_path, "rb") as stateFile:
                self.pyboy.load_state(stateFile)

        self.action_space_buttons = np.array(
            [
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
        )
        self.action_space = np.arange(len(self.action_space_buttons))

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

        self.reset(init=True)


    def random_move(self):
        return np.random.choice(self.action_space)

    def log_info_on_reset(self):
       

        self.runs_data[self.run] = {
            "used_sight": self.use_sight,
            "num_images_seen": self.imgs.num_images(),
            "visited_locations": len(self.locs),
            "visited_xy": len(self.xy),
            "max_total_level": self.max_total_level,
            "max_total_exp": self.max_total_exp,
            "steps": self.steps,
            "rewards_per_location": {k:v for k,v in self.rewards_per_location.items() if len(v) > 0},
            "buttons": self.buttons,
            "state_file": self.state_path,
            "rom_path": self.paths[0],
            "timeout": self.timeout,
            "timeoutcap": self.timeoutcap,
            "run_time": time.time() - self.run_time,
        }

    def is_new_vision(self):
        return self.imgs.check_and_store_image(self.screen_image())[0]

    def write_log(self, filepath):
        if not os.path.isdir(os.path.dirname(filepath)):
            os.mkdir(os.path.dirname(filepath))

        with open(filepath, "w") as f:
            json.dump(self.runs_data, f, indent=4)

    def set_state(self, statefile):
        # check if it exists already in the temp directory
        if os.path.isfile(statefile):
            self.state_path = statefile
        else:
            self.state_path = shutil.copy(statefile, self.temp_dir)

    def reset(self, init=False):
        if init:
            self.imgs = ImageMemory()
            self.run = 0
            self.runs_data = {}
        else:
            self.log_info_on_reset()
            if len(self.locs) > 3:
                print("Found an interesting run, saving!")
                self.imgs.save_all_images(f"./results/good_locs{self.run}")
            self.imgs.reset()
        with open(self.paths[1], "rb") as stateFile:
            self.pyboy.load_state(stateFile)

        self.max_total_level = 0
        self.max_total_exp = 0
        self.locs = set()
        self.xy = set()
        self.reward = 0
        self.button = None
        self.rewards_per_location = {i: [] for i in range(256)}
        self.steps = 0
        self.buttons = []
        self.timeout = self.ogTimeout
        self.run += 1
        self.run_time = time.time()
        self.step(len(self.action_space) - 1)  # pass
        return self.screen_image()

    def save_state(self, file):
        self.pyboy.save_state(file)

    def step(self, movement, ticks_per_input=10, wait=480):
        self.pyboy._rendering(False)
        movement = self.action_space_buttons[movement]
        if movement != "PASS":
            self.pyboy.send_input(self.event_dict_press[movement])
            [self.pyboy.tick() for _ in range(ticks_per_input)]
            self.pyboy.send_input(self.event_dict_release[movement])
        else:
            [self.pyboy.tick() for _ in range(ticks_per_input)]
        [self.pyboy.tick() for _ in range(wait)]
        self.pyboy._rendering(True)
        self.pyboy.tick()
        next_state = self.screen_image()
        self.reward = calc_rewards(self, use_sight=self.use_sight)
        self.rewards_per_location[self.get_current_location()].append(self.reward)
        self.steps += 1
        self.button = movement
        self.buttons.append(movement)
        self.frames_per_loc[self.get_current_location()] = (
            self.frames_per_loc[self.get_current_location()] + 1
        )
        self.done = True if self.steps == self.timeout else False
        return next_state, self.reward, self.done

    def screen_image(self):
        return self.pyboy.botsupport_manager().screen().screen_image()

    def get_frames_in_current_location(self):
        return self.frames_per_loc[self.get_current_location()]

    def extend_timeout(self, time):
        if self.timeout < self.timeoutcap:
            self.timeout += time

    def get_memory_value(self, address):
        return self.pyboy.get_memory_value(address)

    def screen_size(self):
        return self.pyboy.botsupport_manager().screen().screen_ndarray().shape[:2]

    def stop(self, save=True):
        self.pyboy.stop(save)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def get_current_location(self):
        loc = self.pyboy.get_memory_value(RAM_locations.location)
        if loc == 7: # starting zone is also 7 if you leave and come back
            loc = 0
        return loc

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
        return int(total_level), int(total_hp), int(total_exp)

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

    def record(self, ep, e, name):
        document(
            ep,
            self.steps,
            self.screen_image(),
            self.button,
            self.reward,
            self.timeoutcap,
            e,
            name,
            self.get_current_location(),
        )

    def close(self):
        self.pyboy.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        try:
            if not os.path.isdir(os.path.dirname(self.log_path)):
                os.mkdir(os.path.dirname(self.log_path))
        except Exception as e:
            print(e)
        self.write_log(self.log_path)
