import numpy as np

class Rewards:
    def __init__(self, goals=None, N_goals_target=2, max_steps=1000, break_on_goal=True):
        self.max_steps = max_steps
        self.N_goals_target = N_goals_target
        self.break_on_goal = break_on_goal
        self.reset()
        if goals:
            self.set_goals(goals)
        
    def reset(self):
        # ... (keep the existing reset logic)
        self.exploration_decay = 1.0  # New: for decaying exploration reward

    # ... (keep other existing methods)

    def calculate_reward(self, env_vars, button_press):
        self.steps += 1
        total_reward = 0

        # Goal reward
        total_reward += self._goal_reward(env_vars)

        # Exploration reward (now with decay)
        total_reward += self._exploration_reward(env_vars) * self.exploration_decay
        self.exploration_decay *= 0.999  # Decay the exploration reward

        # Pokedex reward (now scaled by rarity)
        total_reward += self._pokedex_reward(env_vars)

        # Step penalty
        total_reward += self._step_penalty()

        # Punish for select and start
        if button_press in ["start", "select"]:
            total_reward -= 0.5

        # Check for episode termination
        if self.done or self.steps >= self.max_steps:
            total_reward += self._episode_end_reward()

        self.cumulative_reward += total_reward

        # Normalize the reward
        normalized_reward = np.clip(total_reward, -5, 5)  # Increased range
        return normalized_reward, self.done

    def _goal_reward(self, env_vars):
        cur_x, cur_y, cur_loc = env_vars["X"], env_vars["Y"], env_vars["map_num_loc"]
        xyl = [cur_x, cur_y, cur_loc]
        for key, value in list(self.reward_goals.items()):
            if xyl in value:
                del self.reward_goals[key]
                self.N_goals += 1
                if self.N_goals >= self.N_goals_target and self.break_on_goal:
                    self.done = True
                # Increased reward for reaching a goal
                return 5.0 * (1 - self.steps / self.max_steps)
        return 0

    def _pokedex_reward(self, env_vars):
        reward = 0
        if env_vars["pkdex_seen"] > self.pkdex_seen:
            reward += 0.2 * self._pokemon_rarity(env_vars["pkdex_seen"])
        self.pkdex_seen = env_vars["pkdex_seen"]
        if env_vars["pkdex_owned"] > self.pkdex_owned:
            reward += 0.6 * self._pokemon_rarity(env_vars["pkdex_owned"])
        self.pkdex_owned = env_vars["pkdex_owned"]
        return reward

    def _pokemon_rarity(self, pokedex_number):
        # Simple rarity scale based on Pokedex number (higher number = rarer)
        return min(1 + pokedex_number / 151, 2)  # Cap at 2x reward

    def _episode_end_reward(self):
        if self.N_goals >= self.N_goals_target:
            # Increased bonus for completing all goals
            return 10.0 * (1 - self.steps / self.max_steps)
        elif self.steps >= self.max_steps:
            return -1.0  # Increased penalty for timeout
        return 0