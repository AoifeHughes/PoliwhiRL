# -*- coding: utf-8 -*-
import numpy as np


class Rewards:
    def __init__(self, goals=None, N_goals_target=2, max_steps=1000):
        self.xy = set()
        self.pkdex_seen = 0
        self.pkdex_owned = 0
        self.money = 0
        self.max_steps = max_steps
        self.total_level = 0
        self.total_hp = 0
        self.total_exp = 0
        self.done = False
        self.reward_goals = {}
        self.reward_goals_rewards = {}
        self.steps = 0
        self.N_goals_target = N_goals_target
        self.goal_completion_times = []
        self.reward_levels = np.linspace(0, 1, 6)  # [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.N_goals = 0
        if goals:
            self.set_goals(goals)

    def set_goals(self, goals):
        self.reward_goals = {}
        self.reward_goals_rewards = {}

        for idx, goal in enumerate(goals):
            self.reward_goals[idx] = []
            for idy, option in enumerate(goal):
                self.reward_goals[idx].append(option[:-1])
                self.reward_goals_rewards[idx] = option[-1]
        self.start_time = self.steps

    def update_for_goals(self, ram):
        cur_x, cur_y, cur_loc = ram["X"], ram["Y"], ram["map_num_loc"]
        xyl = [cur_x, cur_y, cur_loc]

        for key, value in list(self.reward_goals.items()):
            for idx, goal in enumerate(value):
                if xyl == goal:
                    del self.reward_goals[key]
                    time_taken = self.steps - self.start_time

                    time_factor = max(0, (self.max_steps - self.steps) / self.max_steps)
                    reward = self.reward_levels[int(time_factor * 5)]

                    self.N_goals += 1
                    self.goal_completion_times.append(time_taken)

                    if self.N_goals == self.N_goals_target:
                        self.done = True

                    return reward

        return self.reward_levels[0]

    def update_for_party_pokemon(self, ram):
        total_level, total_hp, total_exp = ram["party_info"]
        reward = self.reward_levels[0]
        if total_level > np.sum(self.total_level):
            reward = self.reward_levels[3]
        self.total_level = total_level
        if total_hp > np.sum(self.total_hp):
            reward = max(reward, self.reward_levels[2])
        self.total_hp = total_hp
        if total_exp > np.sum(self.total_exp):
            reward = max(reward, self.reward_levels[2])
        self.total_exp = total_exp
        return reward

    def update_for_movement(self, ram):
        cur_xy = (ram["X"], ram["Y"])
        if cur_xy not in self.xy:
            self.xy.add(cur_xy)
            return self.reward_levels[1]
        return -self.reward_levels[0] * 0.1

    def update_for_pokedex(self, ram):
        if ram["pkdex_seen"] > self.pkdex_seen:
            self.pkdex_seen = ram["pkdex_seen"]
            return self.reward_levels[2]
        if ram["pkdex_owned"] > self.pkdex_owned:
            self.pkdex_owned = ram["pkdex_owned"]
            print("Got a new pokemon!")
            self.done = True
            return self.reward_levels[4]
        return self.reward_levels[0]

    def update_for_money(self, ram):
        if ram["money"] > self.money:
            self.money = ram["money"]
            return self.reward_levels[1]
        elif ram["money"] < self.money:
            self.money = ram["money"]
            return self.reward_levels[0]
        return self.reward_levels[0]

    def calc_rewards(self, env_vars, steps):
        self.steps = steps
        total_reward = self.reward_levels[0]

        for f in [
            self.update_for_party_pokemon,
            self.update_for_movement,
            self.update_for_pokedex,
            self.update_for_money,
            self.update_for_goals,
        ]:
            total_reward = max(total_reward, f(env_vars))

        if self.steps >= self.max_steps:
            self.done = True
            return -self.reward_levels[-1], self.done

        return total_reward, self.done
