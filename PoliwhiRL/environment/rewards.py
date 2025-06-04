# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict, deque, defaultdict, Counter


class Rewards:
    def __init__(self, config):
        # Configuration parameters
        self.max_steps = config["episode_length"]
        self.N_goals_target = config["N_goals_target"]
        self.break_on_goal = config["break_on_goal"]
        self.punish_steps = config["punish_steps"]

        # Rescaled reward values - simplified and positive-focused
        self.small_reward = 0.1
        self.medium_reward = 0.5
        self.large_reward = 1.0
        self.small_penalty = -0.05  # Reduced penalties
        self.medium_penalty = -0.1
        self.large_penalty = -0.5

        # Goal achievement rewards - simple flat rewards
        self.goal_reward = 10.0  # Flat reward for achieving goals
        self.final_goal_bonus = 5.0  # Extra bonus for completing all goals

        # Clipping
        self.clip = 20.0  # Reasonable clipping range

        # Other rewards and penalties - simplified
        self.exploration_reward = 0.2  # Small exploration bonus
        self.step_penalty = -0.01 if self.punish_steps else 0  # Very light step penalty
        self.button_penalty = self.small_penalty  # Reduced menu penalty
        self.pokedex_seen_reward = self.small_reward
        self.pokedex_owned_reward = self.medium_reward
        
        # Simplified exploration parameters
        self.novelty_reward = 0.1  # Small fixed reward for new locations
        self.idle_penalty_threshold = 10  # More lenient idle threshold
        self.idle_penalty = -0.1  # Small idle penalty
        self.map_transition_bonus = 0.5  # Small bonus for new maps

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

        # Variables for ordered goals
        self.location_goals = OrderedDict()
        self.current_goal_index = 0

        # Distance-based guidance reward
        self.distance_reward_factor = 0.1  # Small reward for moving toward goals
        
        # Simplified tracking
        self.last_position = None
        self.idle_counter = 0
        self.discovered_maps = set()

        self.set_goals(config["location_goals"], config["pokedex_goals"])

        if self.N_goals_target == -1:
            self.N_goals_target = len(self.location_goals) + len(self.pokedex_goals)

    def update_targets(self, n_goals_target, max_steps):
        self.N_goals_target = n_goals_target
        self.max_steps = max_steps
        self.done = False

    def set_goals(self, location_goals, pokedex_goals):
        self.location_goals = OrderedDict()
        self.pokedex_goals = {}
        if location_goals:
            for idx, goal in enumerate(location_goals):
                # Handle both list format [[x,y,map,room]] and dict format [{"x":1,"y":2,"map":3}]
                processed_goal = []
                for option in goal:
                    if isinstance(option, dict):
                        # Dict format - extract x, y, map values
                        processed_goal.append(
                            [
                                option.get("x", 0),
                                option.get("y", 0),
                                option.get("map", 0),
                            ]
                        )
                    elif isinstance(option, list):
                        # List format - take first 3 elements (x, y, map)
                        processed_goal.append(option[:3])
                    else:
                        raise ValueError(f"Unknown goal format: {option}")
                self.location_goals[idx] = processed_goal
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
        
        # Check for goal achievements
        goal_reward = self._check_goals(env_vars)
        total_reward += goal_reward
        
        # Simple exploration reward for new locations
        current_location = ((env_vars["X"], env_vars["Y"]), env_vars["map_num"])
        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            total_reward += self.exploration_reward
            
            # Small bonus for new maps
            if env_vars["map_num"] not in self.discovered_maps:
                self.discovered_maps.add(env_vars["map_num"])
                total_reward += self.map_transition_bonus
        
        # Distance guidance toward current goal
        if self.N_goals < self.N_goals_target:
            total_reward += self._distance_based_reward(env_vars)
        
        # Simple movement penalties
        current_position = (env_vars["X"], env_vars["Y"], env_vars["map_num"])
        if self.last_position == current_position:
            self.idle_counter += 1
            if self.idle_counter > self.idle_penalty_threshold:
                total_reward += self.idle_penalty
        else:
            self.idle_counter = 0
        self.last_position = current_position

        # Small penalty for menu buttons
        if button_press in ["start", "select"]:
            total_reward += self.button_penalty

        # Light step penalty
        total_reward += self.step_penalty

        # Check if done
        if self.steps > self.max_steps:
            self.done = True

        self.last_action = button_press
        self.cumulative_reward += total_reward
        clipped_reward = np.clip(total_reward, -self.clip, self.clip).astype(np.float32)

        return clipped_reward, self.done

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

        # Check pokedex goals
        for goal_type in ["seen", "owned"]:
            if env_vars[f"pokedex_{goal_type}"] > getattr(self, f"pokedex_{goal_type}"):
                reward += (
                    self.pokedex_seen_reward
                    if goal_type == "seen"
                    else self.pokedex_owned_reward
                )
                setattr(self, f"pokedex_{goal_type}", env_vars[f"pokedex_{goal_type}"])
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
                
                # Simple flat reward for goal achievement
                reward = self.goal_reward
                
                # Bonus for completing all goals
                if self.N_goals >= self.N_goals_target:
                    reward += self.final_goal_bonus
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
            
            # Simple flat reward
            reward = self.goal_reward
            
            # Bonus for completing all goals
            if self.N_goals >= self.N_goals_target:
                reward += self.final_goal_bonus
                if self.break_on_goal:
                    self.done = True
                    
            return reward
        return 0

    # Remove complex adaptive reward methods - they're no longer needed


    def _distance_based_reward(self, env_vars):
        if self.current_goal_index >= len(self.location_goals):
            return 0
        current_location = (env_vars["X"], env_vars["Y"])
        goal = list(self.location_goals.values())[self.current_goal_index][0]
        if env_vars["map_num"] == goal[2]:  # Check if on the same map
            # Calculate Manhattan distance for simplicity
            distance = abs(current_location[0] - goal[0]) + abs(current_location[1] - goal[1])
            # Small reward that decreases with distance
            return self.distance_reward_factor * max(0, 1 - distance / 50)
        return 0

    def get_progress(self):
        return {
            "Steps": self.steps,
            "Goals Reached": self.N_goals,
            "Pokédex Seen": self.pokedex_seen,
            "Pokédex Owned": self.pokedex_owned,
            "Explored Tiles": len(self.explored_tiles),
            "Discovered Maps": len(self.discovered_maps),
            "Idle Counter": self.idle_counter,
            "Cumulative Reward": self.cumulative_reward,
        }
