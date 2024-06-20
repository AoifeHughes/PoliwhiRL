# Adopted from https://github.com/NicoleFaye/PyBoy/blob/rl-test/PokemonPinballEnv.py
import os
import shutil
from PoliwhiRL.environment import RAM
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
import tempfile
from PoliwhiRL.utils.utils import document
from .rewards import Rewards
from .imagememory import ImageMemory


actions = ['','a', 'b', 'left', 'right', 'up', 'down', 'start'] # select has been removed!



class GenericPyBoyEnv(gym.Env):

    def __init__(self, config, debug=False):
        super().__init__()
        self.config = config
        self.temp_dir = tempfile.mkdtemp()
        self.timeout = config.get("episode_length", 100)
        self.vision = config.get("vision", False)
        self._fitness=0
        self._previous_fitness=0
        self.debug = debug
        self.steps = 0
        self.episode = -1
        self.button = 0
        self.setup_reward_images()
        self.imgs = ImageMemory()
        self.use_grayscale = config.get("use_grayscale", False)
        self.scaling_factor = config.get("scaling_factor", 1)
        self.render = self.vision
 
        self.action_space = spaces.Discrete(len(actions))


        files_to_copy = [config.get("rom_path"), config.get("state_path")]
        files_to_copy.extend(
            [file for file in config.get("extra_files", []) if os.path.isfile(file)]
        )
        self.paths = [shutil.copy(file, self.temp_dir) for file in files_to_copy]
        self.state_path = self.paths[1]
        self.reset_reward_images()

        self.pyboy = PyBoy(self.paths[0], debug=False, window="null")
        self.pyboy.set_emulation_speed(0)
        self.ram = RAM.RAMManagement(self.pyboy)
        self.reset(init=True)
        if not self.debug:
            self.pyboy.set_emulation_speed(0)
        self.reward_calculator = Rewards(self)


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

    def enable_render(self):
        self.render = True

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.button = actions[action]
        # Move the agent
        if action == 0:
            pass
        else:
            self.pyboy.button_press(actions[action])
            self.pyboy.tick(15, False)
            self.pyboy.button_release(actions[action])

        self.pyboy.tick(60, self.render)
        self.steps += 1
        self.done = True if self.steps == self.timeout else False


        self._calculate_fitness()
        reward=self._fitness
        observation=self.pyboy.game_area() if not self.vision else self.screen_image()
        info = {}
        truncated = False

        return observation, reward, self.done, truncated, info

    def screen_size(self):
        return np.array(self.pyboy.screen.image)[
            :, :, :3
        ].shape[:2]

    def set_done(self):
        self.done = True

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness
        self._fitness=self.reward_calculator.calc_rewards(button_pressed=self.button)

    def reset(self, **kwargs):
        self.imgs.reset()
        self.reset_reward_images()
        self.button = 0
        with open(self.state_path, "rb") as stateFile:
            self.pyboy.load_state(stateFile)
        self.reward_calculator = Rewards(self)
        self._fitness=0
        self._previous_fitness=0

        observation=self.pyboy.game_area() if not self.vision else self.screen_image()
        info = {}
        self.steps = 0
        self.episode += 1
        self.render = self.vision


        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()

    def update_RAM_variables(self):
        self.ram.update_variables()

    def reset_reward_images(self):
        self.reward_image_multipliers = {}
        self.reward_image_memory = ImageMemory()
        if (
            "reward_image_folder" in self.config
            and self.config["reward_image_folder"] != ""
        ):
            for img_loc in self.reward_images:
                isAdded, targetHash = self.reward_image_memory.check_and_store_image(
                    img_loc
                )
                if isAdded:
                    image_multiplier = int(img_loc.split("_")[-1][:-4])
                    self.reward_image_multipliers[targetHash] = image_multiplier


    def get_RAM_variables(self):
        return self.ram.get_variables()

    def screen_image(self, no_resize=False):
        original_image = np.array(self.pyboy.screen.image)[
            :, :, :3
        ]  # Remove alpha channel

        if self.use_grayscale:    
            grayscale_image = np.dot(original_image[..., :3], [0.2989, 0.5870, 0.1140])
            grayscale_image = np.expand_dims(grayscale_image, axis=-1)
            return grayscale_image.astype(np.uint8)
        return original_image.astype(np.uint8)
        
    def record(self, fldr):
        document(self.episode,
                self.steps,
                self.screen_image(),
                self.button, 
                self._fitness, 
                self.timeout, 
                0, 
                fldr, 
                0, 
                0, 
                0, 
                False, 
                0
        )

    def get_RAM_variables(self):
        return self.ram.get_variables()
