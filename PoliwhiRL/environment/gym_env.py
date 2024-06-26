# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy
from PoliwhiRL.environment import RAM
from PoliwhiRL.utils.utils import document
from .rewards import Rewards

actions = ["", "a", "b", "left", "right", "up", "down", "start"]  # , 'select']


class PyBoyEnvironment(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temp_dir = tempfile.mkdtemp()
        self._fitness = 0
        self.steps = 0
        self.episode = -1
        self.button = 0
        self.action_space = spaces.Discrete(len(actions))
        self.render = config.get("vision", False)

        files_to_copy = [config.get("rom_path"), config.get("state_path")]
        files_to_copy.extend(
            [file for file in config.get("extra_files", []) if os.path.isfile(file)]
        )
        self.paths = [shutil.copy(file, self.temp_dir) for file in files_to_copy]
        self.state_path = self.paths[1]

        self.pyboy = PyBoy(self.paths[0], window="null")
        self.pyboy.set_emulation_speed(0)
        self.ram = RAM.RAMManagement(self.pyboy)
        self.pyboy.set_emulation_speed(0)
        self.reset()

    def enable_render(self):
        self.render = True

    def handle_action(self, action):
        self.button = actions[action]
        if action != 0:
            self.pyboy.button_press(self.button)
            self.pyboy.tick(15, False)
            self.pyboy.button_release(self.button)
        self.pyboy.tick(75, self.render)
        self.steps += 1
        self.done = self.steps == self.config.get("episode_length", 100)

    def step(self, action):
        self.handle_action(action)
        self._calculate_fitness()
        observation = (
            self.get_game_area()
            if not self.config.get("vision", False)
            else self.get_screen_image()
        )
        return observation, self._fitness, self.done, False, {}

    def get_game_area(self):
        return self.pyboy.game_area()[:18, :20]

    def get_screen_size(self):
        return self.get_screen_image().shape

    def _calculate_fitness(self):
        self._fitness, reward_done = self.reward_calculator.calc_rewards(
            self.get_RAM_variables(), self.steps
        )
        if reward_done:
            self.done = True

    def reset(self):
        self.button = 0
        with open(self.state_path, "rb") as stateFile:
            self.pyboy.load_state(stateFile)
        self.reward_calculator = Rewards(
            self.config.get("reward_goals", None),
            self.config.get("N_goals_target", 2),
            self.config.get("episode_length", 100),
        )
        self._fitness = 0
        self.handle_action(0)
        observation = (
            self.get_game_area()
            if not self.config.get("vision", False)
            else self.get_screen_image()
        )
        self.steps = 0
        self.episode += 1
        self.render = self.config.get("vision", False)
        self._calculate_fitness()
        return observation, {}

    def render(self, mode="human"):
        pass

    def close(self):
        self.pyboy.stop()

    def get_RAM_variables(self):
        return self.ram.get_variables()

    def get_screen_image(self, no_resize=False):
        original_image = np.array(self.pyboy.screen.image)[:, :, :3]

        if self.config.get("use_grayscale", False) and not no_resize:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            original_image = np.expand_dims(original_image, axis=-1)

        if self.config.get("scaling_factor", 1) == 1.0 or no_resize:
            return original_image.astype(np.uint8)
        else:
            new_width = int(
                original_image.shape[1] * self.config.get("scaling_factor", 1)
            )
            new_height = int(
                original_image.shape[0] * self.config.get("scaling_factor", 1)
            )
            resized_image = cv2.resize(
                original_image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            return resized_image.astype(np.uint8)

    def get_pyboy_bg(self):
        return self.pyboy.tilemap_background[:18, :20]

    def get_pyboy_wnd(self):
        return self.pyboy.tilemap_window[:18, :20]

    def record(self, fldr):
        document(
            self.episode,
            self.steps,
            self.get_screen_image(),
            self.button,
            self._fitness,
            fldr,
        )
