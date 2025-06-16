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

        # Multi-objective reward system
        self.use_multi_objective = config.get("use_multi_objective_rewards", True)
        self.reward_weights = {
            'exploration': config.get("exploration_reward_weight", 0.3),
            'story_progress': config.get("story_progress_reward_weight", 0.4),
            'battle_effectiveness': config.get("battle_effectiveness_reward_weight", 0.2),
            'efficiency': config.get("efficiency_reward_weight", 0.1)
        }

        # Rescaled reward values - simplified and positive-focused
        self.small_reward = 0.1
        self.medium_reward = 0.5
        self.large_reward = 1.0
        self.small_penalty = -0.05  # Reduced penalties
        self.medium_penalty = -0.1
        self.large_penalty = -0.5

        # Goal achievement rewards - scale with episode difficulty
        episode_scale = max(1.0, self.max_steps / 1000.0)  # Scale based on episode length
        self.goal_reward = 10.0 * episode_scale  # Scale goal rewards with episode length
        self.final_goal_bonus = 5.0 * episode_scale  # Extra bonus for completing all goals

        # Clipping
        self.clip = 20.0  # Reasonable clipping range

        # Other rewards and penalties - scaled for episode length
        self.exploration_reward = 0.2 * episode_scale  # Scale exploration bonus
        # Dramatically reduce step penalty for long episodes
        step_penalty_scale = min(1.0, 100.0 / self.max_steps)  # Reduce penalty for longer episodes
        self.step_penalty = -0.01 * step_penalty_scale if self.punish_steps else 0
        self.button_penalty = self.small_penalty  # Keep menu penalty small
        self.pokedex_seen_reward = self.small_reward * episode_scale
        self.pokedex_owned_reward = self.medium_reward * episode_scale
        
        # Enhanced exploration parameters
        self.novelty_reward = 0.1 * episode_scale  # Scale novelty reward
        self.idle_penalty_threshold = min(50, max(10, self.max_steps // 100))  # Scale idle threshold
        self.idle_penalty = -0.1 * step_penalty_scale  # Scale idle penalty
        self.map_transition_bonus = 0.5 * episode_scale  # Scale map bonus

        # Multi-objective tracking
        self.objective_scores = {
            'exploration': 0.0,
            'story_progress': 0.0,
            'battle_effectiveness': 0.0,
            'efficiency': 0.0
        }
        self.objective_history = defaultdict(list)
        
        # Battle effectiveness tracking
        self.last_pokemon_hp = {}
        self.battle_encounters = 0
        self.successful_battles = 0
        self.exp_gained = 0
        self.last_exp = 0
        
        # Story progression tracking
        self.story_milestones = set()
        self.significant_items = set()
        self.badge_count = 0
        
        # Efficiency tracking
        self.actions_per_goal = []
        self.current_goal_actions = 0

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
        self.distance_reward_factor = 0.1 * episode_scale  # Scale distance reward
        
        # Milestone system for long episodes
        self.milestone_interval = max(100, self.max_steps // 20)  # Milestone every 5% of episode
        self.milestone_reward = 1.0 * episode_scale  # Reward for reaching milestones
        self.last_milestone = 0
        
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
        self.current_goal_actions += 1
        
        if self.use_multi_objective:
            reward, done = self._calculate_multi_objective_reward(env_vars, button_press)
        else:
            reward, done = self._calculate_single_objective_reward(env_vars, button_press)
        
        self.last_action = button_press
        return reward, done
    
    def _calculate_multi_objective_reward(self, env_vars, button_press):
        """Calculate reward using multi-objective approach."""
        objective_rewards = {}
        
        # 1. Exploration Objective
        exploration_reward = self._calculate_exploration_reward(env_vars)
        objective_rewards['exploration'] = exploration_reward
        
        # 2. Story Progress Objective  
        story_reward = self._calculate_story_progress_reward(env_vars)
        objective_rewards['story_progress'] = story_reward
        
        # 3. Battle Effectiveness Objective
        battle_reward = self._calculate_battle_effectiveness_reward(env_vars)
        objective_rewards['battle_effectiveness'] = battle_reward
        
        # 4. Efficiency Objective
        efficiency_reward = self._calculate_efficiency_reward(env_vars, button_press)
        objective_rewards['efficiency'] = efficiency_reward
        
        # Update objective scores
        for objective, reward in objective_rewards.items():
            self.objective_scores[objective] += reward
            self.objective_history[objective].append(reward)
        
        # Weighted combination of objectives
        total_reward = sum(
            self.reward_weights[obj] * reward 
            for obj, reward in objective_rewards.items()
        )
        
        # Apply curriculum-based objective reweighting
        total_reward = self._apply_curriculum_weighting(total_reward, objective_rewards)
        
        # Check if done
        if self.steps > self.max_steps:
            self.done = True

        # Apply limits and update cumulative
        self.cumulative_reward += total_reward
        clipped_reward = np.clip(total_reward, -self.clip, self.clip).astype(np.float32)
        return clipped_reward, self.done
    
    def _calculate_single_objective_reward(self, env_vars, button_press):
        """Original single-objective reward calculation."""
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
        
        # Milestone rewards for long episodes
        current_milestone = self.steps // self.milestone_interval
        if current_milestone > self.last_milestone:
            total_reward += self.milestone_reward
            self.last_milestone = current_milestone

        # Check if done
        if self.steps > self.max_steps:
            self.done = True

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

    def _calculate_exploration_reward(self, env_vars):
        """Calculate exploration-focused reward."""
        reward = 0
        
        # New location discovery
        current_location = ((env_vars["X"], env_vars["Y"]), env_vars["map_num"])
        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            reward += self.exploration_reward
            
            # Bonus for new maps
            if env_vars["map_num"] not in self.discovered_maps:
                self.discovered_maps.add(env_vars["map_num"])
                reward += self.map_transition_bonus
        
        # Novelty reward based on area coverage
        exploration_density = len(self.explored_tiles) / max(self.steps, 1)
        if exploration_density > 0.01:  # High exploration rate
            reward += self.novelty_reward * exploration_density
            
        return reward
    
    def _calculate_story_progress_reward(self, env_vars):
        """Calculate story progression reward."""
        reward = 0
        
        # Goal achievements (main story progress)
        goal_reward = self._check_goals(env_vars)
        reward += goal_reward * 2.0  # Higher weight for story goals
        
        # Pokedex progress (secondary story element)
        new_pokedex_seen = env_vars.get("pokedex_seen", 0)
        new_pokedex_owned = env_vars.get("pokedex_owned", 0)
        
        if new_pokedex_seen > self.pokedex_seen:
            reward += self.pokedex_seen_reward * (new_pokedex_seen - self.pokedex_seen)
            self.pokedex_seen = new_pokedex_seen
            
        if new_pokedex_owned > self.pokedex_owned:
            reward += self.pokedex_owned_reward * (new_pokedex_owned - self.pokedex_owned)
            self.pokedex_owned = new_pokedex_owned
        
        return reward
    
    def _calculate_battle_effectiveness_reward(self, env_vars):
        """Calculate battle performance reward."""
        reward = 0
        
        # Track party HP and exp changes for battle effectiveness
        party_info = env_vars.get("party_info", (0, 0, 0))
        current_total_level, current_total_hp, current_total_exp = party_info
        
        # Experience gain reward
        if current_total_exp > self.last_exp:
            exp_gain = current_total_exp - self.last_exp
            reward += min(exp_gain / 1000.0, 1.0)  # Scaled exp reward
            self.exp_gained += exp_gain
            
        self.last_exp = current_total_exp
        
        # HP management (penalty for letting Pokemon faint)
        if current_total_hp == 0 and self.last_pokemon_hp.get('total', 1) > 0:
            reward -= 0.5  # Penalty for party wipeout
        
        self.last_pokemon_hp['total'] = current_total_hp
        
        return reward
    
    def _calculate_efficiency_reward(self, env_vars, button_press):
        """Calculate efficiency-based reward."""
        reward = 0
        
        # Movement efficiency
        current_position = (env_vars["X"], env_vars["Y"], env_vars["map_num"])
        if self.last_position == current_position:
            self.idle_counter += 1
            if self.idle_counter > self.idle_penalty_threshold:
                reward += self.idle_penalty * 2  # Stronger idle penalty for efficiency
        else:
            self.idle_counter = 0
            reward += 0.01  # Small reward for movement
        self.last_position = current_position
        
        # Action efficiency penalties
        if button_press in ["start", "select"]:
            reward += self.button_penalty * 1.5  # Stronger menu penalty
            
        # Step efficiency
        reward += self.step_penalty
        
        return reward
    
    def _apply_curriculum_weighting(self, total_reward, objective_rewards):
        """Apply curriculum-based reweighting of objectives."""
        # Adjust weights based on current stage of curriculum
        stage_progress = self.N_goals / max(self.N_goals_target, 1)
        
        # Early stage: focus more on exploration
        if stage_progress < 0.3:
            exploration_boost = 0.2 * objective_rewards['exploration']
            total_reward += exploration_boost
            
        # Mid stage: balance story and efficiency
        elif stage_progress < 0.7:
            story_boost = 0.1 * objective_rewards['story_progress']
            efficiency_boost = 0.1 * objective_rewards['efficiency']
            total_reward += story_boost + efficiency_boost
            
        # Late stage: prioritize completion and efficiency
        else:
            completion_boost = 0.3 * objective_rewards['story_progress']
            efficiency_boost = 0.2 * objective_rewards['efficiency']
            total_reward += completion_boost + efficiency_boost
            
        return total_reward

    def get_progress(self):
        progress = {
            "Steps": self.steps,
            "Goals Reached": self.N_goals,
            "Pokédex Seen": self.pokedex_seen,
            "Pokédex Owned": self.pokedex_owned,
            "Explored Tiles": len(self.explored_tiles),
            "Discovered Maps": len(self.discovered_maps),
            "Idle Counter": self.idle_counter,
            "Cumulative Reward": self.cumulative_reward,
        }
        
        # Add multi-objective scores if enabled
        if self.use_multi_objective:
            progress.update({
                "Objective Scores": self.objective_scores.copy(),
                "Exp Gained": self.exp_gained,
                "Battle Encounters": self.battle_encounters,
                "Current Goal Actions": self.current_goal_actions
            })
            
        return progress
