import numpy as np

class Rewards:
    def __init__(self, goals=None, N_goals_target=2):
        self.xy = set()
        self.pkdex_seen = 0
        self.pkdex_owned = 0
        self.money = 0
        self.total_level = 0
        self.total_hp = 0
        self.total_exp = 0
        self.goal = ''
        self.done = False
        self.reward_goals = {}
        self.reward_goals_rewards = {}
        self.default_reward = 0.1
        self.steps = 0
        self.steps_since_goal = 0
        self.N_goals_target = N_goals_target
        self.N_goals = 0
        if goals:
            self.set_goals(goals)

    def set_goals(self, goals):
        # goals are given as a list of lists
        # but are converted to dict of list of tuples, and rewards dict
        # each tuple is a goal, any of them being met completes that
        # and should then delete the list containing it
        self.reward_goals = {}
        self.reward_goals_rewards = {}

        for idx, goal in enumerate(goals):
            self.reward_goals[idx] = []
            for idy, option in enumerate(goal):
                self.reward_goals[idx].append(option[:-1])
                self.reward_goals_rewards[idx] = option[-1]
        

    def update_for_goals(self, ram):
        reward = 0 
        cur_x = ram['X']
        cur_y = ram['Y']
        cur_loc = ram['map_num_loc']
        xyl = [cur_x, cur_y, cur_loc]

        for key, value in self.reward_goals.items():
            for idx, goal in enumerate(value):
                if xyl == goal:
                    del self.reward_goals[key]
                    reward = self.reward_goals_rewards[key] * (1-(self.steps_since_goal/500)) # todo: come up with better way to time this
                    reward = max(reward, self.default_reward*5)
                    self.steps_since_goal = 0
                    self.N_goals += 1
                    print("Completed goal", key, "reward:", reward)
                    if self.N_goals == self.N_goals_target:
                        print("Completed all required goals")
                        self.done = True
                    return reward
        self.steps_since_goal += 1
        return reward




    def update_for_party_pokemon(self, ram):
        total_level, total_hp, total_exp = ram["party_info"]
        reward = 0
        if total_level > np.sum(self.total_level):
            reward += self.default_reward * 1000
        self.total_level = total_level
        if total_hp > np.sum(self.total_hp):
            reward += self.default_reward * 100
        self.total_hp = total_hp
        if total_exp > np.sum(self.total_exp):
            reward += self.default_reward * 100
        self.total_exp = total_exp
        return reward

    def update_for_movement(self, ram):
        reward = 0
        cur_xy = (ram['X'], ram['Y'])
        if cur_xy not in self.xy:
            reward += self.default_reward * 5
        self.xy.add(cur_xy)
        return reward

    def update_for_pokedex(self, ram):
        reward = 0
        if ram['pkdex_seen'] > self.pkdex_seen:
            reward += self.default_reward * 100
        self.pkdex_seen = ram['pkdex_seen']
        if ram['pkdex_owned'] > self.pkdex_owned:
            self.pkdex_owned = ram['pkdex_owned'] 
            reward += (self.default_reward * 200)
            # set as goal complete
            if self.goal == 'pkmn':
                print("Got a new pokemon!")
                self.done = True
        return reward

    def update_for_money(self, ram):
        reward = 0
        if ram['money'] > self.money:
            reward += self.default_reward * 3
        elif ram['money'] < self.money:
            reward -= self.default_reward * 3
        self.money = ram['money']
        return reward
    

    def calc_rewards(self, env_vars, steps):
        total_reward = -self.default_reward  # negative reward for not doing anything
        self.steps = steps
        for f in [self.update_for_party_pokemon, 
                  self.update_for_movement, 
                  self.update_for_pokedex,
                  self.update_for_money,
                  self.update_for_goals]:
            total_reward += f(env_vars)

        return total_reward, self.done