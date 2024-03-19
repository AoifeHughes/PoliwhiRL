# -*- coding: utf-8 -*-

import os
import shutil
import tempfile

import time
import json
import numpy as np
from pyboy import PyBoy
from . import RAM
from PoliwhiRL.utils import OCR, document
from .rewards import Rewards
from .imagememory import ImageMemory


class Controller:
    def __init__(self, config):
        self.config = config
        self.temp_dir = tempfile.mkdtemp()
        self.ogTimeout = config.get("episode_length", 100)
        self.timeout = self.ogTimeout
        self.timeoutcap = self.ogTimeout * 1000
        self.frames_per_loc = {i: 0 for i in range(256)}
        self.use_sight = config.get("use_sight", False)
        self.scaling_factor = config.get("scaling_factor", 1)
        self.reward_locations_xy = config.get("reward_locations_xy", {})
        self.setup_reward_images()
        self.imgs = ImageMemory()
        self.run = 0
        self.runs_data = {}
        self.reset_reward_images()
        files_to_copy = [config.get("rom_path"), config.get("state_path")]
        files_to_copy.extend(
            [file for file in config.get("extra_files", []) if os.path.isfile(file)]
        )
        self.use_grayscale = config.get("use_grayscale", False)
        self.paths = [shutil.copy(file, self.temp_dir) for file in files_to_copy]
        self.state_path = self.paths[1]
        self.pyboy = PyBoy(self.paths[0], debug=False, window="null")
        self.pyboy.set_emulation_speed(0)
        self.ram = RAM.RAMManagement(self.pyboy)
        self.rewards = Rewards(self)

        self.action_space_buttons = np.array(
            [
                "up",
                "down",
                "left",
                "right",
                "a",
                "b",
                "start",
                "select",
                "pass",
            ]
        )
        self.action_space = np.arange(len(self.action_space_buttons))

        self.reset(init=True)

    def setup_reward_images(self):
        if (
            "reward_image_folder" in self.config
            and self.config["reward_image_folder"] != ""
        ):
            fldr = self.config["reward_image_folder"]
            self.reward_images = [
                f"{fldr}/{f}"
                for f in os.listdir(self.config["reward_image_folder"])
                if f.endswith(".png")
            ]

    def update_RAM_variables(self):
        self.ram.update_variables()

    def get_RAM_variables(self):
        return self.ram.get_variables()

    def reset_reward_images(self):
        self.reward_image_memory = ImageMemory()
        if (
            "reward_image_folder" in self.config
            and self.config["reward_image_folder"] != ""
        ):
            for img_loc in self.reward_images:
                self.reward_image_memory.check_and_store_image(img_loc)

    def reset(self, init=False):

        self.imgs.reset()
        self.reset_reward_images()
        with open(self.state_path, "rb") as stateFile:
            self.pyboy.load_state(stateFile)
        self.save_on_reset = False
        self.button = None
        self.steps = 0
        self.buttons = []
        if not init:
            self.run += 1
        self.run_time = time.time()
        self.done = False
        self.rewards = Rewards(self)
        self.step(len(self.action_space) - 1, init=True)  # pass
        self.timeout = self.ogTimeout
        return self.screen_image()

    def step(self, movement, ticks_per_input=10, wait=75, init=False):
        movement_int = movement
        movement = self.action_space_buttons[movement]
        if movement != "pass":
            self.pyboy.button_press(movement)
            self.pyboy.tick(ticks_per_input, False)
            self.pyboy.button_release(movement)
        else:
            self.pyboy.tick(ticks_per_input, False)
        self.pyboy.tick(wait, True)
        next_state = self.screen_image()
        self.reward = self.rewards.calc_rewards(button_pressed=movement)
        if not init:
            self.steps += 1
            self.button = movement
            self.buttons.append(movement_int)
            self.done = True if self.steps == self.timeout else False
        else:
            self.reward = 0
        return next_state, self.reward, self.done

    def get_buttons(self):
        return self.buttons


    def screen_image(self, no_resize=False):
        original_image = np.array(self.pyboy.screen.image)[
            :, :, :3
        ]  # Remove alpha channel
        if self.use_grayscale:
            grayscale_image = np.dot(original_image[..., :3], [0.2989, 0.5870, 0.1140])
            grayscale_image = np.expand_dims(grayscale_image, axis=-1)
            original_image = grayscale_image
        if self.scaling_factor == 1.0 or no_resize:
            return original_image.astype(np.uint8)
        else:
            original_height, original_width, num_channels = original_image.shape
            new_height = int(original_height * self.scaling_factor)
            new_width = int(original_width * self.scaling_factor)
            resized_image = original_image.reshape(
                new_height,
                original_height // new_height,
                new_width,
                original_width // new_width,
                num_channels,
            ).mean(axis=(1, 3))
            return resized_image.astype(np.uint8)

    def get_frames_in_current_location(self):
        return self.frames_per_loc[self.get_current_location()]

    def extend_timeout(self, time):
        self.timeout += time

    def screen_size(self):
        return self.screen_image().shape[:2]

    def write_log(self, filepath):
        try:
            if not os.path.isdir(os.path.dirname(filepath)):
                os.mkdir(os.path.dirname(filepath))
        except Exception as e:
            print(e)

        with open(filepath, "w") as f:
            json.dump(self.runs_data, f, indent=4)

    def set_state(self, statefile):
        # check if it exists already in the temp directory
        if os.path.isfile(statefile):
            self.state_path = statefile
        else:
            self.state_path = shutil.copy(statefile, self.temp_dir)

    def save_state(self, file):
        self.pyboy.save_state(file)

    def get_text_on_screen(self):
        text = OCR.extract_text(OCR.preprocess_image(self.screen_image()))
        return text

    def store_state(self, state, i):
        # check states folder exists and create if not
        state.seek(0)
        if not os.path.isdir("./states"):
            os.mkdir("./states")
        with open(f"./states/state_{i}.state", "wb") as f:
            f.write(state.read())

    def record(self, e, name, was_random=False, priority_val=0):
        document(
            self.run,
            self.steps,
            self.screen_image(no_resize=True),
            self.button,
            self.reward,
            self.timeoutcap,
            e,
            name,
            self.ram.get_current_location(),
            self.ram.get_XY()[0],
            self.ram.get_XY()[1],
            was_random,
            priority_val,
        )

    def close(self):
        self.pyboy.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
