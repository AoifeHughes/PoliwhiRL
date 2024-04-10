# -*- coding: utf-8 -*-
import numpy as np


class Rewards:
    def __init__(self, controller):
        self.controller = controller
        self.reward_image_multipliers = controller.reward_image_multipliers
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
        self.max_time_in_last_screen = 100
        self.done = False
        self.timeout = controller.timeout

    def update_env_vars(self):
        self.screen = self.controller.screen_image(no_resize=True)
        self.env_vars = self.controller.get_RAM_variables()

    def update_for_party_pokemon(self, total_reward):
        total_level, total_hp, total_exp = self.env_vars["party_info"]
        if total_level > np.sum(self.total_level):
            total_reward += 0.5
            self.total_level = total_level

        if total_hp > np.sum(self.total_hp):
            total_reward += 0.2
            self.total_hp = total_hp

        if total_exp > np.sum(self.total_exp):
            total_reward += 0.2
            self.total_exp = total_exp

        return total_reward

    def update_for_movement(self, total_reward):
        cur_xy = (self.env_vars["X"], self.env_vars["Y"])
        if cur_xy not in self.xy:
            total_reward += 0.01
            self.xy.add(cur_xy)
        return total_reward

    def update_for_image_reward(self, total_reward):
        is_reward_image, img_hash = self.img_rewards.check_if_image_exists(self.screen)
        if is_reward_image:
            self.N_images_rewarded += 1
            total_reward += 1  # * self.reward_image_multipliers[img_hash]
            self.img_rewards.pop_image(img_hash)
            # self.done = True
        return total_reward

    def update_for_pokedex(self, total_reward):
        if self.env_vars["pkdex_seen"] > self.pkdex_seen:
            total_reward += 0.2
            self.pkdex_seen = self.env_vars["pkdex_seen"]

        if self.env_vars["pkdex_owned"] > self.pkdex_owned:
            total_reward += 0.3
            self.pkdex_owned = self.env_vars["pkdex_owned"]
        return total_reward

    def update_for_money(self, total_reward):
        player_money = self.env_vars["money"]
        if player_money > self.money:
            total_reward += 0.1
            self.money = player_money
        elif player_money < self.money:
            total_reward -= 0.1
            self.money = player_money
        return total_reward


    def update_for_timeout(self, total_reward):
        if self.controller.steps >= self.timeout:
            # total_reward -= 0.5
            self.done = True
        return total_reward

    def calc_rewards(self, use_sight=False, button_pressed=None):
        self.update_env_vars()  # Update env_vars at the start
        self.button_pressed = button_pressed
        total_reward = -0.001
        self.done = False

        for func in [
            self.update_for_party_pokemon,
            self.update_for_movement,
            self.update_for_pokedex,
            self.update_for_money,
            self.update_for_image_reward,
            self.update_for_timeout,
        ]:
            total_reward = func(total_reward)

        # Clip the reward to be between -1 and 1
        total_reward = np.clip(total_reward, -1.0, 1.0)

        return total_reward, self.done
