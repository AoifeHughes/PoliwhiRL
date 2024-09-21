# -*- coding: utf-8 -*-
import numpy as np


class Rewards:
    def __init__(self, config):
        # Configuration parameters
        self.max_steps = config["episode_length"]
        self.N_goals_target = config["N_goals_target"]
        self.break_on_goal = config["break_on_goal"]
        self.punish_steps = config["punish_steps"]
        
        # Simplified reward values
        self.small_reward = 0.1
        self.medium_reward = 0.5
        self.large_reward = 1.0
        self.small_penalty = -0.1
        self.medium_penalty = -0.5
        self.large_penalty = -1.0

        # Goal achievement rewards
        self.goal_reward_max = 10.0
        self.goal_reward_min = 1.0

        # Other rewards and penalties
        self.exploration_reward = self.small_reward
        self.step_penalty = self.small_penalty if self.punish_steps else 0
        self.button_penalty = self.medium_penalty
        self.pokedex_seen_reward = self.medium_reward
        self.pokedex_owned_reward = self.large_reward
        
        # State variables
        self.pokedex_seen = 0
        self.pokedex_owned = 0
        self.done = False
        self.last_action = None
        self.steps = 0
        self.N_goals = 0
        self.explored_tiles = set()
        self.last_location = None
        self.cumulative_reward = 0
        self.allowed_pokedex_goals = ["seen", "owned"]

        self.set_goals(config["location_goals"], config["pokedex_goals"])

        if self.N_goals_target == -1:
            self.N_goals_target = len(self.location_goals) + len(self.pokedex_goals)

    def update_targets(self, n_goals_target, max_steps):
        self.N_goals_target = n_goals_target
        self.max_steps = max_steps
        self.done = False

    def set_goals(self, location_goals, pokedex_goals):
        self.location_goals = {}
        self.pokedex_goals = {}
        if location_goals:
            for idx, goal in enumerate(location_goals):
                self.location_goals[idx] = [option[:-1] for option in goal]
        if pokedex_goals:
            if isinstance(pokedex_goals, dict):
                for k, v in pokedex_goals.items():
                    if k in self.allowed_pokedex_goals:
                        self.pokedex_goals[k] = v
            else:
                raise ValueError("Pokedex goals must be a dictionary")

    def calculate_reward(self, env_vars, button_press):
        self.steps += 1
        total_reward = 0

        total_reward += self._check_goals(env_vars)
        total_reward += self._exploration_reward(env_vars)
        total_reward += self._step_penalty()

        if button_press in ["start", "select"]:
            total_reward += self.button_penalty

        self.last_action = button_press

        if self.done or self.steps > self.max_steps:
            self.done = True

        self.cumulative_reward += total_reward

        return np.clip(total_reward, -10, 10), self.done

    def _check_goals(self, env_vars):
        reward = 0

        # Check location goals
        cur_x, cur_y, cur_loc, _ = (
            env_vars["X"],
            env_vars["Y"],
            env_vars["map_num"],
            env_vars["room"],
        )
        xyl = [cur_x, cur_y, cur_loc]
        reward += self._check_goal_achievement(self.location_goals, xyl)

        # Check pokedex goals
        for goal_type in ["seen", "owned"]:
            if env_vars[f"pokedex_{goal_type}"] > getattr(self, f"pokedex_{goal_type}"):
                reward += (
                    self.pokedex_seen_reward
                    if goal_type == "seen"
                    else self.pokedex_owned_reward
                )
                setattr(self, f"pokedex_{goal_type}", env_vars[f"pokedex_{goal_type}"])
                reward += self._check_goal_achievement(
                    self.pokedex_goals, env_vars[f"pokedex_{goal_type}"], goal_type
                )
        return reward

    def _check_goal_achievement(self, goals, current_value, goal_type=None):
        for key, value in list(goals.items()):
            if (goal_type is None and current_value in value) or (
                goal_type and current_value >= value
            ):
                del goals[key]
                self.N_goals += 1
                # Calculate decaying reward
                progress = self.steps / self.max_steps
                reward = self.goal_reward_max - (
                    (self.goal_reward_max - self.goal_reward_min) * progress
                )
                reward = max(self.goal_reward_min, min(self.goal_reward_max, reward))
                if self.N_goals >= self.N_goals_target:
                    if self.break_on_goal:
                        self.done = True
                    return reward
                return reward
        return 0

    def _exploration_reward(self, env_vars):
        current_location = ((env_vars["X"], env_vars["Y"]), env_vars["map_num"])
        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            return self.exploration_reward
        return 0

    def _step_penalty(self):
        return self.step_penalty

    def get_progress(self):
        return {
            "Steps": self.steps,
            "Goals Reached": self.N_goals,
            "Pokédex Seen": self.pokedex_seen,
            "Pokédex Owned": self.pokedex_owned,
            "Explored Tiles": len(self.explored_tiles),
            "Cumulative Reward": self.cumulative_reward,
        }
