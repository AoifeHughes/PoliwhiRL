# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict, deque


class Rewards:
    def __init__(self, config):
        # Core configuration
        self.max_steps = config["episode_length"]
        self.N_goals_target = config["N_goals_target"]
        self.break_on_goal = config["break_on_goal"]
        self.punish_steps = config["punish_steps"]

        # Curriculum parameters
        self.curriculum_level = config.get("curriculum_level", 0)
        self.curriculum_stages = {
            0: "navigation",  # Basic movement and map exploration
            1: "goal_seeking",  # Single goal achievement
            2: "multi_goal",  # Multiple sequential goals
            3: "optimization",  # Efficient path finding
            4: "full_gameplay",  # All mechanics including Pokedex
        }

        # Stage-specific reward scales
        self.reward_scales = {
            "navigation": {
                "exploration": 1.0,
                "goal": 0.5,
                "efficiency": 0.0,
                "pokedex": 0.0,
            },
            "goal_seeking": {
                "exploration": 0.5,
                "goal": 1.0,
                "efficiency": 0.2,
                "pokedex": 0.0,
            },
            "multi_goal": {
                "exploration": 0.3,
                "goal": 1.0,
                "efficiency": 0.5,
                "pokedex": 0.3,
            },
            "optimization": {
                "exploration": 0.2,
                "goal": 1.0,
                "efficiency": 1.0,
                "pokedex": 0.5,
            },
            "full_gameplay": {
                "exploration": 0.2,
                "goal": 1.0,
                "efficiency": 1.0,
                "pokedex": 1.0,
            },
        }

        # Base reward values
        self.base_rewards = {
            "small": 0.2,
            "medium": 0.5,
            "large": 1.0,
            "goal_max": 5.0,
            "goal_min": 1.0,
        }

        # Initialize reward values based on curriculum level
        self._initialize_reward_values()

        # Clipping
        self.clip = 10.0

        # State variables
        self.pokedex_seen = 0
        self.pokedex_owned = 0
        self.done = False
        self.last_action = None
        self.steps = 0
        self.N_goals = 0
        self.explored_tiles = set()
        self.explored_maps = set()
        self.last_location = None
        self.cumulative_reward = 0
        self.consecutive_goals = 0
        self.last_goal_step = 0
        self.allowed_pokedex_goals = ["seen", "owned"]

        # Screen state tracking
        self.screen_history = deque(maxlen=5)
        self.static_state_counter = 0

        # Path tracking
        self.path_history = deque(maxlen=10)

        # Goals setup
        self.location_goals = OrderedDict()
        self.current_goal_index = 0
        self.pokedex_goals = {}

        # Set initial goals
        self.set_goals(config["location_goals"], config["pokedex_goals"])

        if self.N_goals_target == -1:
            self.N_goals_target = len(self.location_goals) + len(self.pokedex_goals)

    def _initialize_reward_values(self):
        """Initialize reward values based on current curriculum stage"""
        stage = self.curriculum_stages[self.curriculum_level]
        scales = self.reward_scales[stage]

        # Scale basic rewards
        self.small_reward = self.base_rewards["small"] * scales["exploration"]
        self.medium_reward = self.base_rewards["medium"] * scales["exploration"]
        self.large_reward = self.base_rewards["large"] * scales["goal"]

        # Scale penalties
        self.small_penalty = -0.1
        self.medium_penalty = -0.5
        self.large_penalty = -1.0

        # Scale goal rewards
        self.goal_reward_max = self.base_rewards["goal_max"] * scales["goal"]
        self.goal_reward_min = self.base_rewards["goal_min"] * scales["goal"]

        # Efficiency rewards scale with curriculum
        self.goal_streak_multiplier = 1.0 + (0.2 * scales["efficiency"])
        self.efficiency_bonus = 2.0 * scales["efficiency"]

        # Other rewards
        self.exploration_reward = self.small_reward
        self.step_penalty = self.small_penalty if self.punish_steps else 0
        self.new_map_reward = self.large_reward
        self.button_penalty = self.medium_penalty
        self.pokedex_seen_reward = self.medium_reward * scales["pokedex"]
        self.pokedex_owned_reward = self.large_reward * scales["pokedex"]
        self.backtrack_penalty = -0.2 * scales["efficiency"]
        self.static_state_penalty = -0.1 * scales["efficiency"]
        self.distance_reward_factor = self.medium_reward * scales["efficiency"]

    def update_curriculum(self, new_level):
        """Update the curriculum level and adjust rewards accordingly"""
        if new_level != self.curriculum_level and new_level in self.curriculum_stages:
            self.curriculum_level = new_level
            self._initialize_reward_values()
            return True
        return False

    def calculate_reward(self, env_vars, button_press, screen_arr):
        self.steps += 1
        total_reward = 0

        # Update screen history and check for static state
        self._update_screen_history(screen_arr)
        static_penalty = self._check_static_state()
        total_reward += static_penalty

        # Core rewards
        goal_reward = self._check_goals(env_vars)
        if goal_reward > 0:
            # Apply streak multiplier for consecutive goals
            goal_reward *= self.goal_streak_multiplier**self.consecutive_goals

            # Add efficiency bonus if goal was reached quickly
            steps_since_last_goal = self.steps - self.last_goal_step
            if steps_since_last_goal < self.max_steps / self.N_goals_target:
                goal_reward += self.efficiency_bonus

            total_reward += goal_reward
            self.consecutive_goals += 1
            self.last_goal_step = self.steps

        # Additional rewards/penalties
        total_reward += self._exploration_reward(env_vars)
        total_reward += self._movement_penalties(env_vars)
        total_reward += self._step_penalty()

        if button_press in ["start", "select"]:
            total_reward += self.button_penalty

        # End-of-episode penalty
        if self.steps > self.max_steps:
            total_reward += self.large_penalty * (self.N_goals_target - self.N_goals)

        self.last_action = button_press
        self._update_state(env_vars)

        if self.done or self.steps >= self.max_steps:
            self.done = True

        self.cumulative_reward += total_reward
        clipped_reward = np.clip(total_reward, -self.clip, self.clip).astype(np.float32)

        return clipped_reward, self.done

    def _update_screen_history(self, screen_arr):
        if len(self.screen_history) == 0 or not np.array_equal(
            screen_arr, self.screen_history[-1]
        ):
            self.screen_history.append(screen_arr)
            self.static_state_counter = 0
        else:
            self.static_state_counter += 1

    def _check_static_state(self):
        if self.static_state_counter > 3:
            return self.static_state_penalty * (self.static_state_counter - 3)
        return 0

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
        reward += self._check_goal_achievement(xyl)

        # Check pokedex goals if curriculum level allows
        if self.curriculum_level >= 2:  # Only check Pokedex in later stages
            for goal_type in ["seen", "owned"]:
                if env_vars[f"pokedex_{goal_type}"] > getattr(
                    self, f"pokedex_{goal_type}"
                ):
                    reward += (
                        self.pokedex_seen_reward
                        if goal_type == "seen"
                        else self.pokedex_owned_reward
                    )
                    setattr(
                        self, f"pokedex_{goal_type}", env_vars[f"pokedex_{goal_type}"]
                    )
                    reward += self._check_pokedex_goal_achievement(
                        env_vars[f"pokedex_{goal_type}"], goal_type
                    )
        return reward

    def _check_goal_achievement(self, current_value):
        if self.current_goal_index < len(self.location_goals):
            goal = list(self.location_goals.values())[self.current_goal_index]
            if current_value in goal:
                self.current_goal_index += 1
                self.N_goals += 1
                progress = self.steps / self.max_steps
                reward = self.goal_reward_max * (1 - progress)
                reward = max(self.goal_reward_min, reward)
                if self.N_goals >= self.N_goals_target:
                    reward *= 2
                    if self.break_on_goal:
                        self.done = True
                return reward
        return 0

    def _check_pokedex_goal_achievement(self, current_value, goal_type):
        if (
            goal_type in self.pokedex_goals
            and current_value >= self.pokedex_goals[goal_type]
        ):
            del self.pokedex_goals[goal_type]
            self.N_goals += 1
            progress = self.steps / self.max_steps
            reward = (
                self.goal_reward_max
                - (self.goal_reward_max - self.goal_reward_min) * progress
            )
            reward = max(self.goal_reward_min, min(self.goal_reward_max, reward))
            if self.N_goals >= self.N_goals_target:
                reward *= 2
                if self.break_on_goal:
                    self.done = True
            return reward
        return 0

    def _exploration_reward(self, env_vars):
        current_location = ((env_vars["X"], env_vars["Y"]), env_vars["map_num"])

        if current_location[1] not in self.explored_maps:
            self.explored_maps.add(current_location[1])
            return self.new_map_reward

        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            dist_reward = self._distance_based_reward(env_vars)
            return self.exploration_reward + dist_reward
        return 0

    def _movement_penalties(self, env_vars):
        current_loc = (env_vars["X"], env_vars["Y"])
        penalty = 0

        if len(self.path_history) >= 2 and current_loc == self.path_history[-2]:
            penalty += self.backtrack_penalty

        return penalty

    def _step_penalty(self):
        step_progress = self.steps / self.max_steps
        dynamic_penalty = self.step_penalty * (1 + step_progress)
        return dynamic_penalty

    def _distance_based_reward(self, env_vars):
        if self.current_goal_index >= len(self.location_goals):
            return 0
        current_location = (env_vars["X"], env_vars["Y"])
        goal = list(self.location_goals.values())[self.current_goal_index][0]
        if env_vars["map_num"] == goal[2]:  # Check if on the same map
            distance = np.sqrt(
                (current_location[0] - goal[0]) ** 2
                + (current_location[1] - goal[1]) ** 2
            )
            return self.distance_reward_factor * (1 / (distance + 1))
        return 0

    def _update_state(self, env_vars):
        current_loc = (env_vars["X"], env_vars["Y"])
        self.path_history.append(current_loc)
        self.last_location = current_loc

    def update_targets(self, n_goals_target, max_steps):
        self.N_goals_target = n_goals_target
        self.max_steps = max_steps
        self.done = False

    def set_goals(self, location_goals, pokedex_goals):
        self.location_goals = OrderedDict()
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

    def get_curriculum_info(self):
        """Get information about current curriculum stage"""
        stage = self.curriculum_stages[self.curriculum_level]
        scales = self.reward_scales[stage]
        return {
            "level": self.curriculum_level,
            "stage": stage,
            "reward_scales": scales,
            "next_stage": self.curriculum_stages.get(
                self.curriculum_level + 1, "max_level"
            ),
        }

    def get_progress(self):
        return {
            "Steps": self.steps,
            "Goals Reached": self.N_goals,
            "Consecutive Goals": self.consecutive_goals,
            "Pokédex Seen": self.pokedex_seen,
            "Pokédex Owned": self.pokedex_owned,
            "Explored Tiles": len(self.explored_tiles),
            "Explored Maps": len(self.explored_maps),
            "Cumulative Reward": self.cumulative_reward,
            "Curriculum Level": self.curriculum_level,
            "Curriculum Stage": self.curriculum_stages[self.curriculum_level],
        }
