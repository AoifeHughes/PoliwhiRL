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
        self.log_path = config.get("log_path", "./logs/log.json")
        self.ogTimeout = config.get("timeout", 100)
        self.timeout = self.ogTimeout
        self.timeoutcap = self.ogTimeout * 1000
        self.frames_per_loc = {i: 0 for i in range(256)}
        self.use_sight = config.get("use_sight", False)
        self.scaling_factor = config.get("scaling_factor", 1)
        self.reward_locations_xy = config.get("reward_locations_xy", {})
        self.state_path = config.get("state_path", "./emu_files/states/start.state")
        self.setup_reward_images()
        self.imgs = ImageMemory()
        self.run = 0
        self.runs_data = {}
        self.reset_reward_images()
        files_to_copy = [config.get("rom_path"), config.get("state_path")]
        files_to_copy.extend([file for file in config.get("extra_files", []) if os.path.isfile(file)])
        self.use_grayscale = config.get("use_grayscale", False)
        self.paths = [shutil.copy(file, self.temp_dir) for file in files_to_copy]
        self.pyboy = PyBoy(config.get("rom_path"), debug=False, window='null')
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
        if 'reward_image_folder' in self.config and self.config['reward_image_folder'] != '':
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
        if 'reward_image_folder' in self.config and self.config['reward_image_folder'] != '':
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
        self.step(len(self.action_space) - 1, init=True)  # pass
        self.timeout = self.ogTimeout
        self.rewards = Rewards(self)
        return self.screen_image()

    def step(self, movement, ticks_per_input=10, wait=60, init=False):
        movement = self.action_space_buttons[movement]
        if movement != "pass":
            self.pyboy.button_press(movement)
            self.pyboy.tick(ticks_per_input, False)
            self.pyboy.button_release(movement)
        else:
            self.pyboy.tick(ticks_per_input, False)
        self.pyboy.tick(wait, True)
        next_state = self.screen_image()
        self.reward = self.rewards.calc_rewards()
        if not init:
            self.steps += 1
            self.button = movement
            self.buttons.append(movement)
            self.done = True if self.steps == self.timeout else False
        else:
            self.reward = 0
        return next_state, self.reward, self.done

    def screen_image(self, no_resize=False):
        # Original image
        original_image = np.array(self.pyboy.screen.image)[:, :, :3]  # Remove alpha channel
        # Convert to grayscale if required
        if self.use_grayscale:
            # Using luminosity method for grayscale conversion
            grayscale_image = np.dot(original_image[..., :3], [0.2989, 0.5870, 0.1140])
            # Expanding dimensions to keep the shape as (height, width, channels)
            grayscale_image = np.expand_dims(grayscale_image, axis=-1)
            # Use the grayscale image as the original image
            original_image = grayscale_image

        # Only resize if scaling_factor is not 1
        if self.scaling_factor == 1.0 or no_resize:
            return original_image.astype(np.uint8)
        else:
            # Calculate new size
            original_height, original_width, num_channels = original_image.shape
            new_height = int(original_height * self.scaling_factor)
            new_width = int(original_width * self.scaling_factor)

            # Reshape and average to downscale the image
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
        if self.timeout < self.timeoutcap:
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

    def set_save_on_reset(self):
        self.save_on_reset = True

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

    def record(self, e, name, was_random=False):
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
