# -*- coding: utf-8 -*-
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


class Rewards:
    def __init__(self, controller):
        self.controller = controller
        self.rewards = []
        self.screen = None
        self.img_memory = controller.imgs
        self.img_rewards = controller.reward_image_memory
        self.xy = set()
        self.env_vars = {}
        self.pkdex_seen = 0
        self.pkdex_owned = 0
        self.money = 0
        self.total_level = 0
        self.total_hp = 0
        self.total_exp = 0
        self.button_pressed = None
        self.N_images_rewarded = 0
        self.locations = set()
        self.last_screen = None
        self.time_in_last_screen = 0

    def update_env_vars(self):
        self.screen = self.controller.screen_image(no_resize=True)
        self.env_vars = self.controller.get_RAM_variables()

    def update_for_vision(self, total_reward, default_reward):
        is_new_vision, _ = self.img_memory.check_and_store_image()
        if is_new_vision:
            total_reward += default_reward * 10
        return total_reward

    def update_for_party_pokemon(self, total_reward, default_reward):
        total_level, total_hp, total_exp = self.env_vars["party_info"]
        if total_level > np.sum(self.total_level):
            total_reward += default_reward * 1000
            self.total_level = total_level

        if total_hp > np.sum(self.total_hp):
            total_reward += default_reward * 100
            self.total_hp = total_hp

        if total_exp > np.sum(self.total_exp):
            total_reward += default_reward * 100
            self.total_exp = total_exp

        return total_reward

    def update_for_movement(self, total_reward, default_reward):
        cur_xy = (self.env_vars["X"], self.env_vars["Y"])
        if cur_xy not in self.xy:
            total_reward += default_reward * 5
            self.xy.add(cur_xy)
        return total_reward

    def update_for_image_reward(self, total_reward, default_reward):
        is_reward_image, img_hash = self.img_rewards.check_if_image_exists(self.screen)
        if is_reward_image:
            self.N_images_rewarded += 1
            total_reward += default_reward * 20  # * self.N_images_rewarded
            self.img_rewards.pop_image(img_hash)
        return total_reward

    def update_for_pokedex(self, total_reward, default_reward):
        if self.env_vars["pkdex_seen"] > self.pkdex_seen:
            total_reward += default_reward * 100
            self.pkdex_seen = self.env_vars["pkdex_seen"]

        if self.env_vars["pkdex_owned"] > self.pkdex_owned:
            total_reward += default_reward * 200
            self.pkdex_owned = self.env_vars["pkdex_owned"]
        return total_reward

    def update_for_money(self, total_reward, default_reward):
        player_money = self.env_vars["money"]
        if player_money > self.money:
            total_reward += default_reward * 3
            self.money = player_money
        elif player_money < self.money:
            total_reward -= default_reward * 3
            self.money = player_money
        return total_reward

    def update_for_menuing(self, total_reward, default_reward):
        if self.button_pressed == "start" or self.button_pressed == "select":
            total_reward += -default_reward * 3
        return total_reward

    def update_for_same_screen(self, total_reward, default_reward):
        if self.last_screen is None:
            self.last_screen = self.controller.screen_image()
            return total_reward

        current_screen = self.controller.screen_image()
        last_screen_gray = cv2.cvtColor(self.last_screen, cv2.COLOR_BGR2GRAY)
        current_screen_gray = cv2.cvtColor(current_screen, cv2.COLOR_BGR2GRAY)
        ssim_index = ssim(last_screen_gray, current_screen_gray)

        if ssim_index >= 0.99:
            self.time_in_last_screen += 1
            total_reward += -default_reward * 2  # * self.time_in_last_screen
        else:
            self.last_screen = current_screen
            self.time_in_last_screen = 0

        return total_reward

    def calc_rewards(self, default_reward=0.01, use_sight=False, button_pressed=None):
        self.update_env_vars()  # Update env_vars at the start
        self.button_pressed = button_pressed
        total_reward = -default_reward  # Penalty for doing nothing
        # if use_sight:
        #     total_reward = self.update_for_vision(total_reward, default_reward)

        for func in [
            self.update_for_party_pokemon,
            self.update_for_movement,
            self.update_for_pokedex,
            self.update_for_money,
            self.update_for_image_reward,
            # self.update_for_menuing,
            # self.update_for_same_screen
        ]:
            total_reward = func(total_reward, default_reward)

        return total_reward
