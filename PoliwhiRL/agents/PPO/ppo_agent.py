# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from collections import deque
from tqdm.auto import tqdm
import torch

from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.utils.visuals import plot_metrics
from PoliwhiRL.replay import PPOMemory
from PoliwhiRL.replay.exploration_memory import ExplorationMemory
from PoliwhiRL.replay.enhanced_exploration_memory import EnhancedExplorationMemory
from PoliwhiRL.models.PPO import PPOModel
from PoliwhiRL.utils.resource_manager import get_resource_pool
from PoliwhiRL.utils.macro_actions import MacroActionLearner


class PPOAgent:
    def __init__(self, input_shape, action_size, config):
        self.config = config
        self.input_shape = input_shape
        self.action_size = action_size
        self.config["input_shape"] = input_shape
        self.config["action_size"] = action_size
        self.device = config["device"]
        
        # Stage-relative episode tracking for curriculum learning - MUST be before config update
        self.stage_episode = 0  # Episode number within current stage
        self.stage_start_episode = 0  # Global episode when stage started
        self.current_n_goals = config.get("N_goals_target", 1)  # Track current stage
        
        # Performance tracking for recovery
        self.performance_window = deque(maxlen=20)  # Track last 20 episodes
        self.performance_baseline = None
        self.degradation_counter = 0
        self.last_recovery_episode = 0
        self.best_checkpoint_path = None
        self.recent_best_reward = float("-inf")  # Best reward in recent window
        
        # Failure learning system
        self.failure_risk_history = deque(maxlen=50)  # Track risk scores
        self.recovery_attempts = []  # Track all intervention attempts
        self.failure_archives = []  # Store detailed failure data
        self.last_checkpoint_restore = -999  # Track when we last restored a checkpoint
        self.training_stats = {  # Clean stats for model evaluation
            "episodes_since_reset": 0,
            "clean_performance": deque(maxlen=100),
            "recovery_points": []
        }
        self.research_stats = {  # Complete session history
            "raw_performance": [],
            "failure_episodes": [],
            "intervention_outcomes": [],
            "risk_scores": []
        }
        
        self.update_parameters_from_config()
        self.best_reward = float("-inf")
        self.best_episode = 0
        self.model = PPOModel(input_shape, action_size, config)
        self.memory = PPOMemory(config)
        # Use enhanced exploration memory if enabled
        use_enhanced = config.get("use_enhanced_exploration_memory", True)
        # Adaptive memory size based on episode length and training mode
        episode_length = config.get("episode_length", 500)
        if episode_length > 10000:  # Long episodes
            memory_size = config.get("exploration_memory_size", 1000)
        else:  # Regular episodes
            memory_size = config.get("exploration_memory_size", 100)
            
        if use_enhanced:
            self.exploration_memory = EnhancedExplorationMemory(
                max_size=memory_size,
                history_length=config.get("ppo_exploration_history_length", 5),
                use_memory=config.get("use_exploration_memory", True),
                action_space_size=action_size,
            )
        else:
            self.exploration_memory = ExplorationMemory(
                max_size=memory_size,
                history_length=config.get("ppo_exploration_history_length", 5),
                use_memory=config.get("use_exploration_memory", True),
            )
        self.reset_tracking()

        # Initialize macro action learning if enabled
        self.use_macro_actions = config.get("use_macro_actions", False)
        if self.use_macro_actions:
            self.macro_learner = MacroActionLearner(
                action_space_size=action_size,
                max_sequence_length=config.get("macro_max_length", 6),
                min_frequency=config.get("macro_min_frequency", 3),
            )
        else:
            self.macro_learner = None

    def update_parameters_from_config(self):
        self.episode = self.config["start_episode"]
        self.record = self.config["record"]
        self.num_episodes = self.config["num_episodes"]
        self.episode_length = self.config["episode_length"]
        self.sequence_length = self.config["sequence_length"]
        self.n_goals = self.config["N_goals_target"]
        self.early_stopping_avg_length = self.config["early_stopping_avg_length"]
        self.record_frequency = self.config["record_frequency"]
        self.results_dir = self.config["results_dir"]
        self.export_state_loc = self.config["export_state_loc"]
        self.extrinsic_reward_weight = self.config["ppo_extrinsic_reward_weight"]
        self.intrinsic_reward_weight = self.config["ppo_intrinsic_reward_weight"]
        
        # Check for stage transition (curriculum learning)
        self._check_stage_transition()
        
    def _check_stage_transition(self):
        """Detect and handle curriculum stage transitions"""
        new_n_goals = self.config.get("N_goals_target", 1)
        
        # Safety check - if current_n_goals not set, this is first initialization
        if not hasattr(self, 'current_n_goals'):
            self.current_n_goals = new_n_goals
            return
            
        if new_n_goals != self.current_n_goals:
            print(f"🎯 Stage transition detected: {self.current_n_goals} → {new_n_goals} goals")
            print(f"   Resetting stage tracking at global episode {self.episode}")
            
            # Reset stage-relative tracking
            self.stage_episode = 0
            self.stage_start_episode = self.episode
            self.current_n_goals = new_n_goals
            
            # Reset performance tracking for new stage
            self.performance_window.clear()
            self.performance_baseline = None
            self.degradation_counter = 0
            self.last_recovery_episode = 0
            
            # Don't reset best checkpoint - it might still be useful
            # But reset recent tracking
            self.recent_best_reward = float("-inf")
            
            # Reset episode data for clean plotting (fresh start for new stage)
            print(f"📊 Clearing episode data for clean stage plotting")
            self.reset_tracking()
            
            print(f"   Stage tracking reset: performance window cleared, counters reset")
            print(f"📈 Episode data cleared - plotting will start fresh for this stage")
        self.checkpoint_frequency = self.config["checkpoint_frequency"]
        self.steps = 0
        self.continue_from_state_loc = self.config["continue_from_state_loc"]
        self.continue_from_state = (
            True if os.path.isfile(self.continue_from_state_loc) else False
        )
        self.train_from_memory = self.config["ppo_train_from_memory"]
        self.report_episode = self.config["report_episode"]
        self.update_frequency = self.config["ppo_update_frequency"]
        self.epochs = self.config["ppo_epochs"]

    def reset_tracking(self):
        self.episode_data = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_losses": [],
            "episode_icm_losses": [],
            "moving_avg_reward": deque(maxlen=100),
            "moving_avg_length": deque(maxlen=100),
            "moving_avg_loss": deque(maxlen=100),
            "moving_avg_icm_loss": deque(maxlen=100),
            "buttons_pressed": deque(maxlen=1000),
            "episode_entropies": [],
        }
        self.episode_data["buttons_pressed"].append(0)
        self.exploration_memory.reset()

        # Reset macro learner episode tracking if available
        if hasattr(self, "macro_learner") and self.macro_learner:
            self.macro_learner._reset_episode()

    def run_curriculum(self, start_goal_n, end_goal_n, step_increment):
        initial_episode_length = self.config["episode_length"]
        for n in range(start_goal_n, end_goal_n + 1):
            self.config["N_goals_target"] = n
            self.config["episode_length"] = initial_episode_length + (
                step_increment * (n - 1)
            )
            self.config["early_stopping_avg_length"] = (
                self.config["episode_length"] // 2
            )
            self.memory.reset(config=self.config)
            self.reset_tracking()
            print(f"Starting training for goal {n}")
            print(f"Episode length: {self.config['episode_length']}")
            print(f"Early stopping length: {self.config['early_stopping_avg_length']}")
            self.update_parameters_from_config()
            self.train_agent()

    def train_agent(self):
        if self.train_from_memory:
            print("Training from memory. Loading data from database and training.")
            self.train_from_memories()

        # Check if we're in a subprocess (multiprocessing context)
        import multiprocessing as mp

        is_subprocess = mp.current_process().name != "MainProcess"

        if self.report_episode and not is_subprocess:
            # Get worker-specific tqdm configuration
            position = self.config.get("tqdm_position", 0)
            desc_prefix = self.config.get("tqdm_desc_prefix", "")

            # Create a positioned tqdm progress bar
            pbar = tqdm(
                range(self.num_episodes),
                desc=f"{desc_prefix} Training (Goals: {self.n_goals})",
                position=position + 1,  # +1 to leave room for the main iteration bar
                leave=False,
            )
        else:
            pbar = range(self.num_episodes)
            if is_subprocess:
                print(
                    f"Agent training in subprocess - {self.num_episodes} episodes, {self.n_goals} goals"
                )
        # Stronger entropy boost for new stages to prevent early collapse
        if self.stage_episode < 50:  # Extended from 30 episodes
            self.model.entropy_boost = 0.1  # Increased from 0.02
        else:
            # Gradual decay instead of sudden drop
            decay_factor = max(0, 1.0 - (self.stage_episode - 50) / 100.0)
            self.model.entropy_boost = 0.1 * decay_factor

        for episode_idx in pbar:
            # Log progress every 5 episodes for hang detection
            if is_subprocess and episode_idx % 5 == 0:
                self._log_training_progress(
                    f"Episode {self.episode}/{self.num_episodes}"
                )

            # Update entropy boost for new stage (same logic as above)
            if self.stage_episode < 50:
                self.model.entropy_boost = 0.1
            else:
                decay_factor = max(0, 1.0 - (self.stage_episode - 50) / 100.0)
                self.model.entropy_boost = 0.1 * decay_factor

            record_loc = (
                f"N_goals_{self.n_goals}/{self.episode}"
                if (self.episode % self.record_frequency == 0 and self.record)
                else None
            )

            if is_subprocess and episode_idx % 5 == 0:
                self._log_training_progress(f"Starting episode {self.episode}")

            self.run_episode(record_loc=record_loc)
            self.episode += 1
            self.stage_episode += 1

            if is_subprocess and episode_idx % 5 == 0:
                self._log_training_progress(f"Completed episode {self.episode - 1}")

            if len(self.memory) > self.sequence_length:
                if is_subprocess and episode_idx % 5 == 0:
                    self._log_training_progress("Updating model")
                self.update_model()

            if self.report_episode and not is_subprocess:
                self._update_progress_bar(pbar)
            if (
                self.episode % 10 == 0 and self.episode > 1
            ) or self.episode == self.num_episodes:
                if is_subprocess:
                    self._log_training_progress("Plotting metrics")
                self._plot_metrics()
            self.model.step_scheduler()

            if (
                self.episode % self.checkpoint_frequency == 0
                and self.config.get("save_checkpoint", True)
                and self.config.get("checkpoint") is not None
            ):
                self.save_model(self.config["checkpoint"])
                
                # Also save periodic safety checkpoints every 50 episodes
                if self.episode % 50 == 0 and self.episode > 0:
                    safety_path = f"{self.config['checkpoint']}_episode_{self.episode}"
                    self.save_model(safety_path)
                    print(f"📌 Saved periodic safety checkpoint at episode {self.episode}")

            # if self._should_stop_early():
            #     break

        if (
            self.config.get("save_checkpoint", True)
            and self.config.get("checkpoint") is not None
        ):
            self.save_model(self.config["checkpoint"])
        # Only save state if explicitly requested
        if self.config.get("save_training_states", False):
            self.run_episode(
                save_path=f"{self.export_state_loc}/N_goals_{self.n_goals}.pkl"
            )
        else:
            self.run_episode(save_path=None)

    def train_from_memories(self):
        memory_ids = self.memory.get_memory_ids(self.config)
        if len(memory_ids) == 0:
            return
        for memory_id in memory_ids:
            data = self.memory.load_from_database(self.config, memory_id)
            self.update_model(data)

    def _should_stop_early(self):
        if (
            np.mean(self.episode_data["moving_avg_length"])
            < self.early_stopping_avg_length
            and self.early_stopping_avg_length > 0
            and self.episode > 110
        ):
            print(
                "Average Steps are below early stopping threshold! Stopping to prevent overfitting."
            )
            return True
        return False

    def run_episode(self, record_loc=None, save_path=None):
        # Use shared temp directory if available in config
        if self.config.get("shared_temp_dir"):
            env = Env(self.config, shared_temp_dir=self.config["shared_temp_dir"])
        else:
            env = Env.create_with_shared_temp(self.config)

        try:
            self.steps = 0
            if self.continue_from_state and (np.random.rand() < 0.9):
                state = env.load_gym_state(
                    self.continue_from_state_loc, self.episode_length, self.n_goals
                )
                self.steps = env.steps
            else:
                state = env.reset()
                self.memory.reset()
                reward_sum = 0
                state_sequence = deque(
                    [state] * self.sequence_length, maxlen=self.sequence_length
                )

                # Update exploration memory with initial screen
                screen = env.get_observation()
                self.exploration_memory.add_screen(screen)
            if record_loc is not None:
                env.enable_record(record_loc, False)

            # Check if we're in a subprocess (multiprocessing context)
            import multiprocessing as mp

            is_subprocess = mp.current_process().name != "MainProcess"

            iter_range = (
                tqdm(
                    range(self.config["episode_length"]),
                    desc=f"{self.config.get('tqdm_desc_prefix', '')} Episode steps",
                    position=self.config.get("tqdm_position", 0)
                    + 2,  # +2 to leave room for iteration and episode bars
                    leave=False,
                )
                if self.report_episode and not is_subprocess
                else range(self.episode_length)
            )

            for _ in iter_range:
                self.steps += 1
                state_seq_arr = np.array(state_sequence)
                # Get exploration memory tensor
                exploration_tensor = self.exploration_memory.get_memory_tensor()
                action = self.model.get_action(state_seq_arr, exploration_tensor)
                self.episode_data["buttons_pressed"].append(action)

                # Log action diversity to file for debugging
                if self.steps % 25 == 0 and self.steps > 0:  # Log every 25 steps
                    self._log_action_diversity_to_file(
                        action, state_seq_arr, exploration_tensor
                    )

                # Track action for macro learning
                if self.macro_learner:
                    state_hash = None
                    if hasattr(self.exploration_memory, "_compute_hash"):
                        state_hash = self.exploration_memory._compute_hash(state)
                    self.macro_learner.add_step(
                        action, 0, state_hash
                    )  # Reward will be updated later

                next_state, extrinsic_reward, done, _ = env.step(action)

                # Update exploration memory with transition information
                if hasattr(self.exploration_memory, "add_transition"):
                    # Enhanced exploration memory with transition tracking
                    coordinates = self._get_coordinates_from_env(env)
                    self.exploration_memory.add_transition(
                        state, action, next_state, extrinsic_reward, coordinates
                    )
                else:
                    # Standard exploration memory
                    self.exploration_memory.add_screen(next_state)

                intrinsic_reward = self.model.compute_intrinsic_reward(
                    state, next_state, action
                )

                # Enhanced exploration bonus computation
                exploration_bonus = self._compute_exploration_bonus(state, next_state, extrinsic_reward)
                
                # Action repetition penalty
                repetition_penalty = 0.0
                recent_actions = list(self.episode_data["buttons_pressed"])[-10:]
                if len(recent_actions) >= 5:
                    # Check if current action has been repeated too much
                    action_count = recent_actions.count(action)
                    if action_count >= 7:  # If 70%+ of recent actions are the same
                        repetition_penalty = 0.1 * action_count  # Increasing penalty
                    
                    # Also check for simple loops (A-B-A-B pattern)
                    if len(recent_actions) >= 4:
                        if (recent_actions[-1] == recent_actions[-3] and 
                            recent_actions[-2] == recent_actions[-4]):
                            repetition_penalty += 0.2  # Additional penalty for loops

                total_reward = self._compute_total_reward(
                    extrinsic_reward - repetition_penalty, intrinsic_reward + exploration_bonus
                )
                reward_sum += extrinsic_reward

                # Update macro learner with reward
                if (
                    self.macro_learner
                    and len(self.macro_learner.current_episode_rewards) > 0
                ):
                    # Update the last step's reward
                    self.macro_learner.current_episode_rewards[-1] = extrinsic_reward

                # Get exploration memory tensor
                exploration_tensor = self.exploration_memory.get_memory_tensor()
                log_prob = self.model.compute_log_prob(
                    state_seq_arr, action, exploration_tensor
                )
                
                # Compute value estimate for GAE
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_seq_arr).unsqueeze(0).to(self.device)
                    exp_tensor = torch.FloatTensor(exploration_tensor).unsqueeze(0).to(self.device) if exploration_tensor is not None else None
                    _, value = self.model.actor_critic(state_tensor, exp_tensor)
                    value = value.item()

                self.memory.store_transition(
                    state,
                    next_state,
                    action,
                    total_reward,
                    done,
                    log_prob,
                    value,
                    exploration_tensor,
                )

                state = next_state
                state_sequence.append(state)

                if done:
                    break

                if (
                    self.steps % self.update_frequency == 0
                    and len(self.memory) > self.sequence_length
                ):
                    self.update_model()

            self._update_episode_stats(reward_sum)

            if save_path is not None:
                env.save_gym_state(save_path)
        finally:
            # Ensure environment is properly closed
            env.close()

    def _compute_total_reward(self, extrinsic_reward, intrinsic_reward):
        return (
            self.extrinsic_reward_weight * extrinsic_reward
            + self.intrinsic_reward_weight * intrinsic_reward
        )
    
    def _compute_exploration_bonus(self, state, next_state, extrinsic_reward):
        """Enhanced exploration bonus computation"""
        exploration_bonus = 0.0
        
        if hasattr(self.exploration_memory, "get_exploration_bonus"):
            state_hash = self.exploration_memory._compute_hash(state)
            
            # MUCH stronger base exploration bonus
            base_bonus = self.exploration_memory.get_exploration_bonus(state_hash)
            exploration_bonus = base_bonus * 0.5  # Increased from 0.1
            
            # Waypoint discovery bonus
            if hasattr(self.exploration_memory, 'waypoints') and state_hash in self.exploration_memory.waypoints:
                waypoint_info = self.exploration_memory.waypoints[state_hash]
                if waypoint_info["visits"] == 1:  # First discovery
                    exploration_bonus += 3.0  # Increased from 2.0
            
            # Repetition penalty - check recent actions
            recent_actions = list(self.episode_data["buttons_pressed"])[-20:]
            if len(recent_actions) >= 10:
                # Count consecutive same actions
                consecutive_count = 1
                for i in range(len(recent_actions)-1, 0, -1):
                    if recent_actions[i] == recent_actions[i-1]:
                        consecutive_count += 1
                    else:
                        break
                
                # Apply penalty for repetitive behavior
                if consecutive_count >= 5:
                    repetition_penalty = min(0.9, consecutive_count * 0.1)
                    exploration_bonus *= (1.0 - repetition_penalty)
            
            # Early stage bonus - stronger exploration incentive
            if self.stage_episode < 100:
                stage_multiplier = 2.0 - (self.stage_episode / 100.0)  # 2x to 1x over 100 episodes
                exploration_bonus *= stage_multiplier
        
        return exploration_bonus

    def _update_episode_stats(self, total_reward):
        self.episode_data["episode_rewards"].append(total_reward)
        self.episode_data["episode_lengths"].append(self.steps)
        self.episode_data["moving_avg_reward"].append(total_reward)
        self.episode_data["moving_avg_length"].append(self.steps)
        
        # Update performance tracking
        self.performance_window.append(total_reward)
        
        # Check if this is the best reward overall
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_episode = self.episode
            # Save best checkpoint
            self._save_best_checkpoint()
            
        # Update recent best
        if len(self.performance_window) > 0:
            self.recent_best_reward = max(self.performance_window)
            
        # Check for severe performance drop only
        if len(self.performance_window) >= 20:
            recent_avg = np.mean(list(self.performance_window)[-10:])
            older_avg = np.mean(list(self.performance_window)[-20:-10])
            
            # Only intervene if performance dropped by more than 50% AND we're stuck
            if recent_avg < older_avg * 0.5 and self.stage_episode > 50:
                # Also check if we're making no progress (rewards too similar)
                reward_variance = np.var(list(self.performance_window)[-10:])
                if reward_variance < 0.01:  # Very low variance = stuck
                    self.degradation_counter += 1
                    if self.degradation_counter > 15:  # Increased threshold
                        self._trigger_recovery()
                else:
                    # Reset counter if there's still some variance (agent is trying)
                    self.degradation_counter = 0
            else:
                self.degradation_counter = max(0, self.degradation_counter - 1)

        # Track current entropy coefficient
        current_entropy = self.model._get_entropy_coef(self.stage_episode)
        self.episode_data["episode_entropies"].append(current_entropy)
        
        # Update dual statistics tracking
        self.research_stats["raw_performance"].append(total_reward)
        self.training_stats["episodes_since_reset"] += 1
        
        # Only add to clean performance if we haven't had recent major interventions
        if len(self.recovery_attempts) == 0 or self.stage_episode - self.recovery_attempts[-1]["stage_episode"] > 10:
            self.training_stats["clean_performance"].append(total_reward)
            
        # Save failure learning stats periodically
        if self.episode % 50 == 0:
            self.save_failure_learning_stats()

        # Process macro actions at end of episode
        if self.macro_learner:
            self.macro_learner.end_episode()

            # Print macro action statistics occasionally
            if self.episode % 50 == 0 and self.episode > 0:
                stats = self.macro_learner.get_macro_statistics()
                if stats.get("num_macros", 0) > 0:
                    print(
                        f"Episode {self.episode}: Discovered {stats['num_macros']} macro actions, "
                        f"avg reward: {stats['avg_macro_reward']:.3f}"
                    )
                    top_macros = self.macro_learner.get_top_macros(3)
                    for i, (sequence, info) in enumerate(top_macros):
                        print(
                            f"  Top {i + 1}: {sequence} (reward: {info['avg_reward']:.3f})"
                        )

    def update_model(self, data=None):
        total_loss = 0
        total_icm_loss = 0
        was_data_none = data is None
        # Pass stage_episode to model for curriculum-aware entropy decay
        self.model.stage_episode = self.stage_episode
        for _ in range(self.epochs):
            data = self.memory.get_all_data() if data is None else data
            if data is None:
                return
            loss, icm_loss = self.model.update(data, self.episode)
            total_loss += loss
            total_icm_loss += icm_loss
        if was_data_none:
            self._update_loss_stats(total_loss, total_icm_loss)
        self.memory.reset()

    def _update_loss_stats(self, total_loss, total_icm_loss):
        steps_since_update = len(self.memory)
        avg_loss = total_loss / (self.epochs * steps_since_update)
        avg_icm_loss = total_icm_loss / (self.epochs * steps_since_update)
        self.episode_data["episode_losses"].append(avg_loss)
        self.episode_data["episode_icm_losses"].append(avg_icm_loss)
        self.episode_data["moving_avg_loss"].append(avg_loss)
        self.episode_data["moving_avg_icm_loss"].append(avg_icm_loss)

    def _update_progress_bar(self, pbar):
        avg_reward = (
            np.mean(self.episode_data["moving_avg_reward"])
            if self.episode_data["moving_avg_reward"]
            else 0
        )
        avg_length = (
            np.mean(self.episode_data["moving_avg_length"])
            if self.episode_data["moving_avg_length"]
            else 0
        )

        current_reward = (
            self.episode_data["episode_rewards"][-1]
            if self.episode_data["episode_rewards"]
            else 0
        )
        current_length = (
            self.episode_data["episode_lengths"][-1]
            if self.episode_data["episode_lengths"]
            else 0
        )

        # Add worker ID to postfix if available
        worker_id = self.config.get("tqdm_worker_id", None)
        postfix = {
            "Avg Reward": f"{avg_reward:.2f}",
            "Avg Length": f"{avg_length:.2f}",
            "Reward": f"{current_reward:.2f}",
            "Length": f"{current_length}",
        }

        if worker_id is not None:
            postfix["Worker"] = worker_id

        pbar.set_postfix(postfix)

    def _plot_metrics(self):
        # Check if this agent has individual agent data (it's part of a multi-agent setup)
        if "individual_agent_data" in self.episode_data:
            # This is the averaged agent with individual agent data
            # First plot the averaged agent's metrics
            avg_title_prefix = "Averaged Agent"
            if hasattr(self, 'current_n_goals') and self.current_n_goals:
                stage_info = f"Stage {self.current_n_goals} ({self.n_goals} goals)"
                avg_title_prefix = f"{avg_title_prefix} - {stage_info}"
            
            plot_metrics(
                self.episode_data["episode_rewards"],
                self.episode_data["episode_losses"],
                self.episode_data["episode_lengths"],
                self.episode_data["buttons_pressed"],
                self.n_goals,
                self.stage_episode if hasattr(self, 'stage_episode') else self.episode,  # Use stage episode for x-axis
                save_loc=self.results_dir,
                title_prefix=avg_title_prefix,
                entropies=self.episode_data.get("episode_entropies", None),
            )

            # Then plot each individual agent's metrics to their respective results directories
            for i, agent_data in enumerate(self.episode_data["individual_agent_data"]):
                # Construct the agent-specific results directory
                agent_results_dir = (
                    f"{self.config['results_dir'].rstrip('_0123456789')}_{i}"
                )

                agent_title_prefix = f"Agent {i}"
                if hasattr(self, 'current_n_goals') and self.current_n_goals:
                    stage_info = f"Stage {self.current_n_goals} ({self.n_goals} goals)"
                    agent_title_prefix = f"{agent_title_prefix} - {stage_info}"
                
                plot_metrics(
                    agent_data["episode_rewards"],
                    agent_data["episode_losses"],
                    agent_data["episode_lengths"],
                    agent_data["buttons_pressed"],
                    self.n_goals,
                    self.stage_episode if hasattr(self, 'stage_episode') else self.episode,  # Use stage episode for x-axis
                    save_loc=agent_results_dir,
                    title_prefix=agent_title_prefix,
                    entropies=agent_data.get("episode_entropies", None),
                )
        else:
            # Get worker ID if available for title prefix
            worker_id = self.config.get("tqdm_worker_id")
            title_prefix = f"Agent {worker_id}" if worker_id is not None else None

            # Ensure the results directory exists
            os.makedirs(self.results_dir, exist_ok=True)

            # This is a regular agent or an individual agent in a multi-agent setup
            # Include stage information in title for curriculum learning
            if hasattr(self, 'current_n_goals') and self.current_n_goals:
                stage_info = f"Stage {self.current_n_goals} ({self.n_goals} goals)"
                if title_prefix:
                    title_prefix = f"{title_prefix} - {stage_info}"
                else:
                    title_prefix = stage_info
            
            plot_metrics(
                self.episode_data["episode_rewards"],
                self.episode_data["episode_losses"],
                self.episode_data["episode_lengths"],
                self.episode_data["buttons_pressed"],
                self.n_goals,
                self.stage_episode if hasattr(self, 'stage_episode') else self.episode,  # Use stage episode for x-axis
                save_loc=self.results_dir,
                title_prefix=title_prefix,
                entropies=self.episode_data.get("episode_entropies", None),
            )

    def save_model(self, path):
        path = f"{path}"
        os.makedirs(path, exist_ok=True)

        resource_pool = get_resource_pool()

        # Use file locking for safe checkpoint writing
        with resource_pool.file_lock(path):
            self.model.save(path)

            # Save additional information including stage tracking
            info = {
                "episode": self.episode,
                "best_reward": self.best_reward,
                "best_episode": self.best_episode,
                "episode_data": self.episode_data,
                "performance_window": list(self.performance_window),
                "performance_baseline": self.performance_baseline,
                "best_checkpoint_path": self.best_checkpoint_path,
                # CURRICULUM FIX: Save stage information for proper transition detection
                "current_n_goals": self.current_n_goals,
                "stage_episode": self.stage_episode,
                "stage_start_episode": self.stage_start_episode,
            }
            torch.save(info, f"{path}/info.pth")

            # Save exploration memory if enhanced
            if hasattr(self.exploration_memory, "save_to_file"):
                try:
                    self.exploration_memory.save_to_file(
                        f"{path}/exploration_memory.pkl"
                    )
                except Exception as e:
                    print(f"Warning: Could not save exploration memory: {e}")

    def load_model(self, path):
        try:
            resource_pool = get_resource_pool()

            # Use file locking for safe checkpoint reading
            with resource_pool.file_lock(path):
                self.model.load(f"{path}")
                torch.serialization.add_safe_globals(["numpy", "np"])
                info = torch.load(
                    f"{path}/info.pth", map_location=self.device, weights_only=False
                )
                self.config["start_episode"] = info["episode"]
                self.episode = info["episode"]
                
                # Restore performance tracking attributes if available
                if "best_reward" in info:
                    self.best_reward = info["best_reward"]
                if "best_episode" in info:
                    self.best_episode = info["best_episode"]
                if "performance_window" in info:
                    self.performance_window = deque(info["performance_window"], maxlen=20)
                if "performance_baseline" in info:
                    self.performance_baseline = info["performance_baseline"]
                if "best_checkpoint_path" in info:
                    self.best_checkpoint_path = info["best_checkpoint_path"]
                
                # CURRICULUM FIX: Restore stage tracking for proper transition detection
                if "current_n_goals" in info:
                    self.current_n_goals = info["current_n_goals"]
                if "stage_episode" in info:
                    self.stage_episode = info["stage_episode"]
                if "stage_start_episode" in info:
                    self.stage_start_episode = info["stage_start_episode"]

                # Debug logging
                worker_id = self.config.get("tqdm_worker_id", "?")
                print(
                    f"Agent {worker_id} loaded checkpoint from {path}, episode {self.episode}"
                )
                print(f"📊 Restored stage info: {self.current_n_goals} goals, stage episode {self.stage_episode}")

                # Load episode data if available, but ensure all required keys exist
                loaded_episode_data = info.get("episode_data", {})
                if loaded_episode_data:
                    # Start with fresh episode data to ensure all keys exist
                    fresh_episode_data = {
                        "episode_rewards": [],
                        "episode_lengths": [],
                        "episode_losses": [],
                        "episode_icm_losses": [],
                        "moving_avg_reward": deque(maxlen=100),
                        "moving_avg_length": deque(maxlen=100),
                        "moving_avg_loss": deque(maxlen=100),
                        "moving_avg_icm_loss": deque(maxlen=100),
                        "buttons_pressed": deque(maxlen=1000),
                        "episode_entropies": [],
                    }
                    # Update with loaded data, preserving any existing values
                    for key, value in loaded_episode_data.items():
                        if key in fresh_episode_data:
                            if isinstance(
                                fresh_episode_data[key], deque
                            ) and not isinstance(value, deque):
                                # Convert list to deque if needed
                                fresh_episode_data[key] = deque(value, maxlen=100)
                            else:
                                fresh_episode_data[key] = value
                    self.episode_data = fresh_episode_data
                    # Ensure buttons_pressed has at least one element
                    if len(self.episode_data["buttons_pressed"]) == 0:
                        self.episode_data["buttons_pressed"].append(0)

                # Load exploration memory if enhanced
                if hasattr(self.exploration_memory, "load_from_file"):
                    try:
                        memory_path = f"{path}/exploration_memory.pkl"
                        if os.path.exists(memory_path):
                            self.exploration_memory.load_from_file(memory_path)
                            print(
                                f"Agent {worker_id} loaded exploration memory from {memory_path}"
                            )
                    except Exception as e:
                        print(f"Warning: Could not load exploration memory: {e}")
                
                # CURRICULUM FIX: Detect stage transition and reset learning parameters
                self._detect_and_handle_stage_transition()

        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")

    def _detect_and_handle_stage_transition(self):
        """Detect if we've transitioned to a new curriculum stage and reset learning parameters"""
        current_goals = self.config.get("N_goals_target", 1)
        
        print(f"🔍 Stage transition check: current_config_goals={current_goals}, saved_goals={self.current_n_goals}")
        
        # Check if this is a new stage (goals increased)
        if current_goals > self.current_n_goals:
            stage_difficulty = current_goals - 1  # 0-indexed difficulty
            print(f"🔄 CURRICULUM STAGE TRANSITION DETECTED: {self.current_n_goals} → {current_goals} goals")
            print(f"📈 Global episode: {self.episode}, Previous stage episodes: {self.stage_episode}")
            
            # Reset learning rate with stage-appropriate boost
            self.model.reset_learning_rate_for_stage(stage_difficulty_multiplier=stage_difficulty)
            
            # Reset stage-specific tracking
            self.stage_episode = 0
            self.stage_start_episode = self.episode
            self.current_n_goals = current_goals
            
            # Add SIGNIFICANT entropy boost for exploration
            # Much higher boost to encourage exploration in new stages
            entropy_boost = 0.1 + (stage_difficulty * 0.05)  # Was 0.02 + 0.01*difficulty
            self.model.entropy_boost = entropy_boost
            print(f"🎯 Applied entropy boost: +{entropy_boost:.3f} for initial exploration")
            
            # Reset performance tracking for clean transition
            self.performance_window.clear()
            self.degradation_counter = 0
            
            # Reset episode data for clean plotting (fresh start for new stage)
            print(f"📊 Clearing episode data for clean stage plotting")
            self.reset_tracking()
            
            print(f"✅ STAGE TRANSITION COMPLETE - ready for {current_goals} goal challenge!")
            print(f"🏁 Stage episode counter reset to 0, global episode continues at {self.episode}")
            print(f"📈 Episode data cleared - plotting will start fresh for this stage")
        else:
            print(f"ℹ️  No stage transition detected (goals unchanged: {current_goals})")

    def _log_action_diversity_to_file(self, action, state_seq_arr, exploration_tensor):
        """Log action diversity metrics to file for debugging agent behavior"""
        try:
            # Create logs directory for this agent
            agent_id = self.config.get("tqdm_worker_id", "main")
            logs_dir = f"{self.results_dir}/action_logs"
            os.makedirs(logs_dir, exist_ok=True)

            log_file = f"{logs_dir}/agent_{agent_id}_actions.log"

            # Get current action probabilities from model
            state_tensor = torch.FloatTensor(state_seq_arr).unsqueeze(0).to(self.device)
            if exploration_tensor is not None:
                exploration_tensor = (
                    torch.FloatTensor(exploration_tensor).unsqueeze(0).to(self.device)
                )

            with torch.no_grad():
                action_probs, _ = self.model.actor_critic(
                    state_tensor, exploration_tensor
                )
                action_probs = action_probs.squeeze().cpu().numpy()

            # Calculate action diversity metrics
            recent_actions = list(self.episode_data["buttons_pressed"])[
                -25:
            ]  # Last 25 actions
            action_counts = {}
            for a in recent_actions:
                action_counts[a] = action_counts.get(a, 0) + 1

            # Calculate entropy of recent actions
            total_recent = len(recent_actions)
            action_entropy = 0.0
            if total_recent > 0:
                for count in action_counts.values():
                    prob = count / total_recent
                    if prob > 0:
                        action_entropy -= prob * np.log2(prob)

            # Calculate entropy of current action probabilities
            prob_entropy = 0.0
            for prob in action_probs:
                if prob > 1e-10:
                    prob_entropy -= prob * np.log2(prob)

            # Get current entropy coefficient
            current_entropy_coef = self.model._get_entropy_coef(self.stage_episode)

            # Map actions to button names for readability
            button_names = ["A", "B", "Right", "Left", "Up", "Down", "Start", "Select"]
            action_name = (
                button_names[action]
                if action < len(button_names)
                else f"Action_{action}"
            )

            # Most frequent recent action
            most_frequent_action = (
                max(action_counts.items(), key=lambda x: x[1])[0]
                if action_counts
                else action
            )
            most_frequent_name = (
                button_names[most_frequent_action]
                if most_frequent_action < len(button_names)
                else f"Action_{most_frequent_action}"
            )
            most_frequent_pct = (
                (action_counts.get(most_frequent_action, 0) / total_recent * 100)
                if total_recent > 0
                else 0
            )

            # Warning flags for critical thresholds
            warning_flags = []
            if action_entropy < 0.5:
                warning_flags.append("LOW_ENTROPY")
            if prob_entropy < 0.5:
                warning_flags.append("OVERCONFIDENT")
            if most_frequent_pct > 85:
                warning_flags.append("ACTION_COLLAPSE")

            warnings_str = (
                f" | WARNINGS: {','.join(warning_flags)}" if warning_flags else ""
            )

            # Create log entry
            log_entry = (
                f"Episode: {self.episode:4d} | Step: {self.steps:4d} | "
                f"Action: {action_name:6s} | "
                f"ActionEntropy: {action_entropy:.3f} | "
                f"ProbEntropy: {prob_entropy:.3f} | "
                f"EntropyCoef: {current_entropy_coef:.4f} | "
                f"MostFreq: {most_frequent_name:6s} ({most_frequent_pct:.1f}%) | "
                f"ActionProbs: [{', '.join([f'{p:.3f}' for p in action_probs])}] | "
                f"RecentActions: {recent_actions[-10:]}{warnings_str}\n"  # Last 10 actions for context
            )

            # Write to file
            with open(log_file, "a") as f:
                f.write(log_entry)

        except Exception as e:
            # Don't let logging errors crash training
            print(f"Warning: Could not log action diversity: {e}")

    def _log_training_progress(self, message):
        """Log training progress for hang detection"""
        try:
            import time

            agent_id = self.config.get("tqdm_worker_id", "main")
            hang_log_dir = f"{self.results_dir}/hang_logs"
            os.makedirs(hang_log_dir, exist_ok=True)
            hang_log_file = f"{hang_log_dir}/agent_{agent_id}_hang_detection.log"

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(hang_log_file, "a") as f:
                f.write(f"[{timestamp}] Agent {agent_id} - PROGRESS: {message}\n")
        except Exception:
            pass  # Don't let logging errors interfere with training

    def _get_coordinates_from_env(self, env):
        """Extract coordinates from environment if available"""
        try:
            if hasattr(env, "ram") and hasattr(env.ram, "get_variables"):
                variables = env.ram.get_variables()
                return (
                    variables.get("X"),
                    variables.get("Y"),
                    variables.get("map_num"),
                )
        except Exception:
            pass
        return None

    def get_episode_data(self):
        return self.episode_data

    def set_episode_data(self, data):
        self.episode_data = data

    def _save_best_checkpoint(self):
        """Save a special checkpoint when achieving best performance"""
        try:
            best_path = f"{self.config['checkpoint']}_best"
            os.makedirs(best_path, exist_ok=True)
            self.save_model(best_path)
            self.best_checkpoint_path = best_path
            print(f"💾 Saved new best checkpoint at episode {self.episode} with reward {self.best_reward:.2f}")
        except Exception as e:
            print(f"Warning: Could not save best checkpoint: {e}")
            
    def _check_performance_degradation_OLD(self):
        """Check if performance is degrading and trigger recovery if needed"""
        if not self.config.get("enable_recovery", True):
            return  # Recovery disabled
            
        if len(self.performance_window) < 10:
            return  # Need enough data
            
        # Calculate failure risk and attempt graduated intervention
        risk_score = self._calculate_failure_risk()
        intervention = self._graduated_intervention(risk_score)
        
        if intervention:
            print(f"🔧 Intervention applied: {intervention}")
            return  # Intervention handled the situation
        
        # Continue with legacy recovery logic as fallback
            
        # Calculate average of recent performance
        recent_avg = np.mean(list(self.performance_window)[-5:])
        window_avg = np.mean(list(self.performance_window))
        
        # Set baseline after initial episodes
        if self.performance_baseline is None and len(self.performance_window) >= 15:
            self.performance_baseline = np.mean(list(self.performance_window)[:10])
            
        if self.performance_baseline is not None:
            # Check for significant degradation
            degradation_threshold = self.config.get("recovery_degradation_threshold", 0.7)
            recovery_threshold = self.config.get("recovery_trigger_threshold", 0.5)
            
            if recent_avg < self.performance_baseline * degradation_threshold:
                self.degradation_counter += 1
                
                # Log degradation warning
                if self.degradation_counter % 5 == 0:
                    print(f"⚠️  Performance degradation detected at episode {self.episode}: "
                          f"recent avg: {recent_avg:.2f}, baseline: {self.performance_baseline:.2f}")
                
                # Trigger recovery if severe degradation
                if (recent_avg < self.performance_baseline * recovery_threshold and 
                    self.stage_episode - self.last_recovery_episode > 20):  # Allow more frequent recovery when failing badly
                    self._trigger_recovery()
            else:
                # Reset degradation counter if performance improves
                if self.degradation_counter > 0:
                    self.degradation_counter = 0  # Full reset when performance improves
                    
            # Update baseline if we've improved significantly
            if self.performance_baseline is not None and window_avg > self.performance_baseline * 1.1:  # 10% improvement
                self.performance_baseline = window_avg
                
        # Simple periodic logging every 25 episodes
        if self.episode % 25 == 0 and self.episode > 0:
            recent_avg = np.mean(list(self.performance_window)[-10:]) if len(self.performance_window) >= 10 else 0.0
            current_entropy = self.model._get_entropy_coef(self.stage_episode)
            lr = self.model.optimizer.param_groups[0]['lr']
            print(f"Episode {self.episode}: avg_reward={recent_avg:.2f}, entropy={current_entropy:.4f}, lr={lr:.6f}")
                
    def _trigger_recovery(self):
        """Simple recovery - just load best checkpoint if available"""
        if self.best_checkpoint_path and self.stage_episode - self.last_recovery_episode > 50:
            print(f"Loading best checkpoint from episode {self.best_episode}")
            try:
                # Save current episode tracking
                current_episode = self.episode
                current_stage = self.stage_episode
                
                # Load checkpoint
                self.load_model(self.best_checkpoint_path)
                
                # Restore episode tracking
                self.episode = current_episode
                self.stage_episode = current_stage
                self.last_recovery_episode = self.stage_episode
                self.degradation_counter = 0
                
                # Small entropy boost after recovery
                self.model.entropy_boost = 0.03
            except Exception as e:
                print(f"Recovery failed: {e}")
        
    def _select_recovery_method_OLD(self):
        """Intelligently select the best recovery method based on current situation"""
        # Check if manual override is set
        if "recovery_method" in self.config and self.config["recovery_method"] != "auto":
            return self.config.get("recovery_method", "best_checkpoint")
            
        # Analyze the situation
        episodes_since_best = self.stage_episode - (self.best_episode - self.stage_start_episode)
        episodes_since_best = max(0, episodes_since_best)  # Ensure non-negative
        recent_avg = np.mean(list(self.performance_window)[-5:]) if len(self.performance_window) >= 5 else 0
        
        # Calculate entropy trend
        recent_entropies = self.episode_data.get("episode_entropies", [])[-10:]
        entropy_declining = False
        if len(recent_entropies) >= 5:
            entropy_trend = np.polyfit(range(len(recent_entropies)), recent_entropies, 1)[0]
            entropy_declining = entropy_trend < -0.001  # Significant decline
        
        # Check for exploration deadlock (high entropy but low novelty)
        novelty_rate = 0.0
        if hasattr(self.exploration_memory, "hash_visits"):
            recent_visits = list(self.exploration_memory.hash_visits.values())[-100:]
            avg_visits = np.mean(recent_visits) if recent_visits else 1.0
            novelty_rate = 1.0 / avg_visits if avg_visits > 0 else 1.0
            
        current_lr = self.model.optimizer.param_groups[0]['lr']
        initial_lr = self.config.get("ppo_learning_rate", 0.0003)
        final_entropy = self.model._get_entropy_coef(self.episode) if hasattr(self.model, '_get_entropy_coef') else 0.0
        
        # Decision logic
        if novelty_rate < 0.05 and final_entropy > 0.15 and current_lr < initial_lr * 0.5:
            # Exploration deadlock: high entropy but stuck in small state space with low LR
            print(f"   → Selected: lr_adjustment (exploration deadlock: high entropy {final_entropy:.3f}, low novelty {novelty_rate:.3f}, low LR {current_lr:.6f})")
            return "lr_adjustment"
            
        elif episodes_since_best < 30 and self.best_checkpoint_path:
            # Recent peak performance - load best checkpoint
            print(f"   → Selected: best_checkpoint (recent peak {episodes_since_best} episodes ago)")
            return "best_checkpoint"
            
        elif entropy_declining and hasattr(self.model.actor_critic, 'reset_actor'):
            # Policy collapsed (low entropy) - partial reset to explore again
            print(f"   → Selected: partial_reset (entropy declining, current: {recent_entropies[-1] if recent_entropies else 0:.4f})")
            return "partial_reset"
            
        elif self.degradation_counter > 15:
            # Long-term struggle - try partial reset to break out of local minima
            print(f"   → Selected: partial_reset (prolonged degradation, counter: {self.degradation_counter})")
            return "partial_reset"
            
        else:
            # Default to best checkpoint if available
            if self.best_checkpoint_path:
                print(f"   → Selected: best_checkpoint (default)")
                return "best_checkpoint"
            else:
                print(f"   → Selected: partial_reset (no checkpoint available)")
                return "partial_reset"
    
    def _partial_model_reset(self):
        """Reset only actor network to recover from bad policy"""
        try:
            # Check if the model has the reset_actor method
            if hasattr(self.model.actor_critic, 'reset_actor'):
                self.model.actor_critic.reset_actor()
                print("   ✓ Actor network reset while preserving critic")
                
                # Also boost entropy after partial reset
                self.model.entropy_boost = 0.25  # Higher boost for partial reset
                print("   ✓ Entropy boosted to 0.25 to encourage exploration")
                
                # Reset degradation counter
                self.degradation_counter = 0
                self.last_recovery_episode = self.stage_episode
                
            else:
                print("   Partial reset not available for this model architecture")
        except Exception as e:
            print(f"   Partial reset failed: {e}")
            
    def _adjust_learning_rate(self, factor=0.5):
        """Temporarily adjust learning rate"""
        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] *= factor
        print(f"   Reduced learning rate by factor of {factor}")
        
    def _update_entropy_boost_OLD(self):
        """Adaptively adjust entropy based on exploration needs"""
        # Calculate exploration metrics
        if hasattr(self.exploration_memory, "hash_visits"):
            # Get recent exploration statistics
            total_states = len(self.exploration_memory.hash_visits)
            recent_visits = list(self.exploration_memory.hash_visits.values())[-100:]
            avg_visits = np.mean(recent_visits) if recent_visits else 1.0

            # Check if we're stuck (visiting same states repeatedly)
            novelty_rate = 1.0 / avg_visits if avg_visits > 0 else 1.0

            # Boost entropy if:
            # 1. Starting a new curriculum stage (low episode count)
            # 2. Low novelty rate (stuck in local patterns)
            # 3. Haven't reached goal recently

            is_new_stage = self.stage_episode < 50  # Extended initial exploration period (was 20)
            is_low_novelty = novelty_rate < 0.5  # More lenient novelty threshold (was 0.3)

            # Check recent goal progress - more sensitive to struggling
            recent_lengths = list(self.episode_data["moving_avg_length"])[-10:]
            is_struggling = (
                len(recent_lengths) > 5
                and np.mean(recent_lengths) > self.episode_length * 0.6  # Reduced from 0.8
            )

            # Calculate entropy boost with much stronger values
            boost = 0.0
            if is_new_stage:
                boost += 0.15  # 15% boost for new curriculum stage (was 5%)
            if is_low_novelty:
                # Check if we're in exploration deadlock (high entropy but low novelty)
                current_entropy = self.model._get_entropy_coef(self.stage_episode) if hasattr(self.model, '_get_entropy_coef') else 0.0
                if current_entropy > 0.2 and novelty_rate < 0.05:
                    # Too much randomness, reduce boost
                    boost += 0.05  # Reduced boost when already very random
                    print(f"   Reducing entropy boost due to exploration deadlock (entropy: {current_entropy:.3f}, novelty: {novelty_rate:.3f})")
                else:
                    boost += 0.20  # 20% boost for low exploration (was 3%)
            if is_struggling:
                boost += 0.10  # 10% boost if struggling to reach goals (was 2%)

            # Much gentler decay - only decay after significant episodes
            # Keep strong exploration for longer, especially when struggling
            decay_factor = max(0.5, 0.99 ** (max(0, self.stage_episode - 100) // 30))
            
            # Extra boost during critical early learning phase
            if self.stage_episode < 100:
                decay_factor = max(decay_factor, 0.8)
                
            self.model.entropy_boost = boost * decay_factor

            # Log entropy adjustments
            if boost > 0 and self.episode % 10 == 0:
                print(
                    f"Episode {self.episode}: Entropy boost = {self.model.entropy_boost:.4f} "
                    f"(new_stage={is_new_stage}, low_novelty={is_low_novelty}, struggling={is_struggling}, "
                    f"total_states={total_states})"
                )
                
            # Comprehensive debugging statistics every 15 episodes or during critical moments
            if (self.episode % 15 == 0 or 
                self.episode in [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75] or  # More frequent critical episodes
                self.degradation_counter > 3):  # Earlier warning threshold
                self._log_debug_statistics(is_new_stage, is_low_novelty, is_struggling, total_states, boost, decay_factor)
        else:
            # No exploration memory, use simple new stage detection with stronger boost
            if self.stage_episode < 100:
                decay_factor = max(0.5, 0.99 ** (max(0, self.stage_episode - 50) // 30))
                self.model.entropy_boost = 0.15 * decay_factor
            else:
                self.model.entropy_boost = 0.08  # Higher minimum baseline boost
                
            # Debug logging for non-exploration memory case
            if (self.stage_episode % 15 == 0 or 
                self.stage_episode in [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75] or 
                self.degradation_counter > 3):
                boost = 0.15 if self.stage_episode < 100 else 0.08
                decay_factor = max(0.5, 0.99 ** (max(0, self.stage_episode - 50) // 30)) if self.stage_episode < 100 else 1.0
                self._log_debug_statistics(self.stage_episode < 100, False, False, 0, boost, decay_factor)
                
    def _log_debug_statistics(self, is_new_stage=False, is_low_novelty=False, is_struggling=False, total_states=0, boost=0.0, decay_factor=1.0):
        """Log comprehensive debugging statistics for analysis"""
        try:
            # Get current entropy components
            base_entropy = max(
                self.model.entropy_coef * self.model.entropy_decay**self.stage_episode, 
                self.model.entropy_min
            ) if hasattr(self.model, 'entropy_coef') else 0.0
            
            entropy_boost = getattr(self.model, "entropy_boost", 0.0)
            final_entropy = self.model._get_entropy_coef(self.stage_episode) if hasattr(self.model, '_get_entropy_coef') else 0.0
            
            # Get recent performance metrics
            recent_rewards = list(self.performance_window)[-10:] if len(self.performance_window) >= 10 else list(self.performance_window)
            recent_avg = np.mean(recent_rewards) if recent_rewards else 0.0
            recent_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
            
            # Get learning rate
            current_lr = self.model.optimizer.param_groups[0]['lr'] if hasattr(self.model, 'optimizer') else 0.0
            
            # Get recent episode lengths
            recent_lengths = list(self.episode_data.get("moving_avg_length", []))[-10:]
            avg_length = np.mean(recent_lengths) if recent_lengths else 0.0
            
            # Get action diversity if available
            recent_entropies = self.episode_data.get("episode_entropies", [])[-10:]
            avg_episode_entropy = np.mean(recent_entropies) if recent_entropies else 0.0
            
            # Exploration memory stats
            novelty_rate = 0.0
            if hasattr(self.exploration_memory, "hash_visits"):
                recent_visits = list(self.exploration_memory.hash_visits.values())[-100:]
                avg_visits = np.mean(recent_visits) if recent_visits else 1.0
                novelty_rate = 1.0 / avg_visits if avg_visits > 0 else 1.0
            
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG STATS - Episode {self.episode} (Stage {self.current_n_goals}, Stage Episode {self.stage_episode})")
            print(f"{'='*80}")
            # Calculate current risk score
            current_risk = self._calculate_failure_risk()
            
            print(f"📊 PERFORMANCE:")
            print(f"   Recent reward avg:     {recent_avg:8.2f} (std: {recent_std:.2f})")
            print(f"   Baseline reward:       {self.performance_baseline:8.2f}" if self.performance_baseline else "   Baseline reward:       Not set")
            print(f"   Recent length avg:     {avg_length:8.1f} / {self.episode_length}")
            print(f"   Degradation counter:   {self.degradation_counter}")
            print(f"   Episodes since best:   {self.episode - self.best_episode}")
            print(f"   Best reward:           {self.best_reward:8.2f}")
            print(f"   🚨 FAILURE RISK:       {current_risk:8.3f} {'⚠️ HIGH' if current_risk > 0.7 else '✅ OK' if current_risk < 0.4 else '🟡 WATCH'}")
            print()
            print(f"🎲 ENTROPY ANALYSIS:")
            print(f"   Base entropy:          {base_entropy:8.4f}")
            print(f"   Entropy boost:         {entropy_boost:8.4f}")
            print(f"   Final entropy coef:    {final_entropy:8.4f}")
            print(f"   Initial entropy:       {self.model.entropy_coef:8.4f}")
            print(f"   Entropy floor (30%):   {self.model.entropy_coef * 0.3:8.4f}")
            print(f"   Boost raw:             {boost:8.4f}")
            print(f"   Decay factor:          {decay_factor:8.4f}")
            print(f"   Avg episode entropy:   {avg_episode_entropy:8.4f}")
            print()
            print(f"🚩 CONDITIONS:")
            print(f"   Is new stage:          {is_new_stage}")
            print(f"   Is low novelty:        {is_low_novelty} (rate: {novelty_rate:.3f})")
            print(f"   Is struggling:         {is_struggling}")
            print(f"   Total states visited:  {total_states}")
            print()
            print(f"⚙️  OPTIMIZATION:")
            print(f"   Learning rate:         {current_lr:8.6f}")
            print(f"   Last recovery episode: {self.last_recovery_episode}")
            print(f"   Episodes since recovery: {self.stage_episode - self.last_recovery_episode}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"Debug statistics error: {e}")
            # Basic fallback info
            print(f"🔍 BASIC DEBUG - Episode {self.episode}: entropy_boost={getattr(self.model, 'entropy_boost', 0.0):.4f}, degradation_counter={self.degradation_counter}")
    
    def _calculate_failure_risk_OLD(self):
        """Calculate failure risk score (0.0 = healthy, 1.0 = critical failure)"""
        if len(self.performance_window) < 10:
            return 0.0  # Not enough data
            
        risk_factors = {}
        
        try:
            # 1. Performance slope (how quickly performance is degrading)
            recent_rewards = list(self.performance_window)[-10:]
            if len(recent_rewards) >= 5:
                x = np.arange(len(recent_rewards))
                slope = np.polyfit(x, recent_rewards, 1)[0]
                # Normalize slope - steep negative = high risk
                risk_factors["performance_slope"] = max(0, -slope / 50.0)  # Adjust scale as needed
            else:
                risk_factors["performance_slope"] = 0.0
                
            # 2. Performance variance (inconsistent = higher risk)
            perf_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
            # Very low std (like 0.00) indicates complete stagnation
            if perf_std < 1.0:
                risk_factors["stagnation"] = 1.0  # Critical stagnation
            else:
                risk_factors["stagnation"] = max(0, 1.0 - perf_std / 100.0)
                
            # 3. Entropy instability (high entropy but poor performance = deadlock)
            current_entropy = self.model._get_entropy_coef(self.stage_episode) if hasattr(self.model, '_get_entropy_coef') else 0.0
            recent_avg = np.mean(recent_rewards)
            baseline = self.performance_baseline if self.performance_baseline else 0.0
            
            if current_entropy > 0.2 and recent_avg < baseline * 0.5:
                risk_factors["entropy_deadlock"] = min(1.0, current_entropy)  # High entropy + poor performance
            else:
                risk_factors["entropy_deadlock"] = 0.0
                
            # 4. Exploration stagnation
            novelty_rate = 0.0
            if hasattr(self.exploration_memory, "hash_visits"):
                recent_visits = list(self.exploration_memory.hash_visits.values())[-100:]
                avg_visits = np.mean(recent_visits) if recent_visits else 1.0
                novelty_rate = 1.0 / avg_visits if avg_visits > 0 else 1.0
            risk_factors["exploration_stagnation"] = max(0, 1.0 - novelty_rate * 20)  # Low novelty = high risk
            
            # 5. Learning rate collapse
            current_lr = self.model.optimizer.param_groups[0]['lr'] if hasattr(self.model, 'optimizer') else 0.0
            initial_lr = self.config.get("ppo_learning_rate", 0.0003)
            lr_ratio = current_lr / initial_lr if initial_lr > 0 else 1.0
            risk_factors["lr_collapse"] = max(0, 1.0 - lr_ratio * 5)  # Very low LR = high risk
            
            # 6. Episodes since improvement (more sensitive)
            episodes_since_best = self.stage_episode - (self.best_episode - self.stage_start_episode)
            episodes_since_best = max(0, episodes_since_best)
            risk_factors["stagnant_episodes"] = min(1.0, episodes_since_best / 50.0)  # 50+ episodes = max risk (was 100)
            
            # 7. Performance drop severity (new factor)
            if self.performance_baseline:
                recent_avg = np.mean(recent_rewards)
                performance_ratio = recent_avg / self.performance_baseline if self.performance_baseline > 0 else 1.0
                # If performance dropped significantly below baseline
                if performance_ratio < 0.5:
                    risk_factors["severe_drop"] = min(1.0, (0.5 - performance_ratio) * 2.0)
                else:
                    risk_factors["severe_drop"] = 0.0
            else:
                risk_factors["severe_drop"] = 0.0
            
            # Weighted combination of risk factors (rebalanced for earlier detection)
            weights = {
                "performance_slope": 0.15,
                "stagnation": 0.20,        # Still critical - complete stagnation
                "entropy_deadlock": 0.20,   
                "exploration_stagnation": 0.15,
                "lr_collapse": 0.10,
                "stagnant_episodes": 0.15,  # Higher weight for episodes since best
                "severe_drop": 0.05         # New factor for sudden drops
            }
            
            total_risk = sum(risk_factors[factor] * weights[factor] for factor in weights if factor in risk_factors)
            total_risk = min(1.0, total_risk)  # Cap at 1.0
            
            # Store for analysis
            self.failure_risk_history.append(total_risk)
            self.research_stats["risk_scores"].append({
                "episode": self.episode,
                "stage_episode": self.stage_episode,
                "total_risk": total_risk,
                "factors": risk_factors.copy()
            })
            
            return total_risk
            
        except Exception as e:
            print(f"Error calculating failure risk: {e}")
            return 0.0
    
    def _graduated_intervention_OLD(self, risk_score):
        """Graduated intervention system based on failure risk"""
        intervention_taken = None
        
        # Get current situation analysis
        episodes_degrading = self.degradation_counter
        episodes_since_best = self.stage_episode - (self.best_episode - self.stage_start_episode)
        episodes_since_best = max(0, episodes_since_best)
        
        try:
            # LEVEL 1: Early Warning (Risk > 0.25) - Much more aggressive!
            if risk_score > 0.25 and episodes_degrading >= 3:
                print(f"⚠️  Early warning detected (risk: {risk_score:.3f})")
                # Preemptive entropy adjustment
                current_entropy = self.model._get_entropy_coef(self.stage_episode)
                if current_entropy > 0.20:  # Lower threshold
                    # Reduce entropy if too high
                    self.model.entropy_boost = max(0.03, self.model.entropy_boost * 0.6)  # More aggressive reduction
                    intervention_taken = "preemptive_entropy_reduction"
                elif current_entropy < 0.08:  # Lower threshold
                    # Increase entropy if too low
                    self.model.entropy_boost = min(0.15, self.model.entropy_boost + 0.08)  # Bigger boost
                    intervention_taken = "preemptive_entropy_increase"
                    
            # LEVEL 2: Mild Failure (Risk > 0.35 OR 10+ degrading episodes OR 30+ since best)
            elif risk_score > 0.35 or episodes_degrading >= 10 or episodes_since_best >= 30:
                print(f"🟡 Mild failure detected (risk: {risk_score:.3f}, degrading: {episodes_degrading}, since_best: {episodes_since_best})")
                # Learning rate boost + entropy adjustment
                current_lr = self.model.optimizer.param_groups[0]['lr']
                initial_lr = self.config.get("ppo_learning_rate", 0.0003)
                if current_lr < initial_lr * 0.7:  # Higher threshold
                    boost_factor = initial_lr * 0.8 / current_lr  # Bigger boost
                    self._adjust_learning_rate(factor=boost_factor)
                    intervention_taken = "learning_rate_boost"
                    print(f"   Boosted learning rate by {boost_factor:.2f}x")
                else:
                    # If LR is fine, adjust entropy instead
                    current_entropy = self.model._get_entropy_coef(self.stage_episode)
                    if current_entropy > 0.25:
                        self.model.entropy_boost = max(0.05, self.model.entropy_boost * 0.5)
                        intervention_taken = "entropy_adjustment"
                        print(f"   Reduced entropy boost")
                
            # LEVEL 3: Moderate Failure (Risk > 0.5 OR 20+ degrading episodes OR 50+ since best)
            elif risk_score > 0.5 or episodes_degrading >= 20 or episodes_since_best >= 50:
                print(f"🟠 Moderate failure detected (risk: {risk_score:.3f}, degrading: {episodes_degrading}, since_best: {episodes_since_best})")
                # Partial reset (actor network) - much earlier!
                if hasattr(self.model.actor_critic, 'reset_actor'):
                    self._partial_model_reset()
                    intervention_taken = "partial_reset"
                    print(f"   Performed partial network reset")
                else:
                    # Fallback to aggressive learning rate adjustment
                    self._adjust_learning_rate(factor=3.0)  # Bigger boost
                    intervention_taken = "lr_fallback"
                    
            # LEVEL 4: Severe Failure (Risk > 0.65 OR 30+ degrading episodes OR 75+ since best)
            elif (risk_score > 0.65 or episodes_degrading >= 30 or episodes_since_best >= 75) and self.stage_episode - self.last_checkpoint_restore > 25:
                print(f"🔴 Severe failure detected (risk: {risk_score:.3f}, degrading: {episodes_degrading}, since_best: {episodes_since_best})")
                # Archive current state before major intervention
                self._archive_failure_state("severe_failure", risk_score)
                
                # Restore best checkpoint if available
                if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                    print(f"   Restoring best checkpoint from episode {self.best_episode}")
                    # Save the best episode before loading (will be overwritten)
                    saved_best_episode = self.best_episode
                    saved_best_reward = self.best_reward
                    self.load_model(self.best_checkpoint_path)
                    
                    # Restore the best tracking (don't let load_model overwrite)
                    self.best_episode = saved_best_episode  
                    self.best_reward = saved_best_reward
                    self.last_checkpoint_restore = self.stage_episode  # Prevent immediate re-trigger
                    
                    # Reset clean training stats
                    self.training_stats["episodes_since_reset"] = 0
                    self.training_stats["clean_performance"].clear()
                    self.training_stats["recovery_points"].append(self.stage_episode)
                    intervention_taken = "checkpoint_restore"
                else:
                    # No checkpoint - perform full reset
                    self._full_reset()
                    intervention_taken = "full_reset"
                    
            # LEVEL 5: Critical Failure (Risk > 0.80 OR 50+ degrading episodes OR 100+ since best)
            elif risk_score > 0.80 or episodes_degrading >= 50 or episodes_since_best >= 100:
                print(f"💀 CRITICAL failure detected (risk: {risk_score:.3f}, degrading: {episodes_degrading}, since_best: {episodes_since_best})")
                # Archive failure and start fresh
                self._archive_failure_state("critical_failure", risk_score)
                
                # Complete restart
                print(f"   Performing complete restart - this run is archived")
                self._fresh_start()
                intervention_taken = "fresh_start"
                
            # Record intervention attempt
            if intervention_taken:
                attempt = {
                    "episode": self.episode,
                    "stage_episode": self.stage_episode,
                    "risk_score": risk_score,
                    "intervention": intervention_taken,
                    "degrading_episodes": episodes_degrading,
                    "episodes_since_best": episodes_since_best,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.recovery_attempts.append(attempt)
                self.research_stats["intervention_outcomes"].append(attempt)
                
                # Reset counters after intervention
                self.degradation_counter = 0
                self.last_recovery_episode = self.stage_episode
                
                return intervention_taken
                
        except Exception as e:
            print(f"Error in graduated intervention: {e}")
            
        return None
    
    def _archive_failure_state(self, failure_type, risk_score):
        """Archive detailed failure data before major intervention"""
        try:
            failure_data = {
                "failure_type": failure_type,
                "episode": self.episode,
                "stage_episode": self.stage_episode,
                "risk_score": risk_score,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "performance_window": list(self.performance_window),
                "degradation_counter": self.degradation_counter,
                "episodes_since_best": self.stage_episode - (self.best_episode - self.stage_start_episode),
                "current_entropy": self.model._get_entropy_coef(self.stage_episode) if hasattr(self.model, '_get_entropy_coef') else 0.0,
                "learning_rate": self.model.optimizer.param_groups[0]['lr'] if hasattr(self.model, 'optimizer') else 0.0,
                "exploration_states": len(self.exploration_memory.hash_visits) if hasattr(self.exploration_memory, 'hash_visits') else 0,
                "recent_risk_scores": list(self.failure_risk_history)[-10:],
                "recovery_attempts": self.recovery_attempts[-5:],  # Last 5 attempts
            }
            
            self.failure_archives.append(failure_data)
            self.research_stats["failure_episodes"].append(failure_data)
            
            # Save to file for persistence
            archive_file = f"{self.results_dir}/failure_archive_{self.episode}.json"
            try:
                import json
                with open(archive_file, 'w') as f:
                    json.dump(failure_data, f, indent=2)
                print(f"   Failure state archived to {archive_file}")
            except Exception as e:
                print(f"   Warning: Could not save archive file: {e}")
                
        except Exception as e:
            print(f"Error archiving failure state: {e}")
    
    def _full_reset(self):
        """Perform full model reset while preserving archives"""
        print(f"   Performing full model reset")
        # Reset model weights
        if hasattr(self.model.actor_critic, 'reset_actor'):
            self.model.actor_critic.reset_actor()
        if hasattr(self.model.actor_critic, 'reset_critic'):
            self.model.actor_critic.reset_critic()
        
        # Reset optimizer
        initial_lr = self.config.get("ppo_learning_rate", 0.0003)
        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] = initial_lr
            
        # Reset entropy
        self.model.entropy_boost = 0.0
        
        # Reset exploration memory
        self.exploration_memory.reset()
        
        # Reset tracking (but keep archives)
        self.performance_window.clear()
        self.performance_baseline = None
        self.degradation_counter = 0
        
    def _fresh_start(self):
        """Complete restart - reset everything except archives"""
        self._full_reset()
        
        # Reset episode tracking within stage
        # NOTE: Don't reset global episode or stage episode - maintain continuity
        
        # Reset best tracking for this "sub-run"
        self.best_reward = float("-inf")
        self.recent_best_reward = float("-inf")
        
        # Clear training stats but keep research data
        self.training_stats["episodes_since_reset"] = 0
        self.training_stats["clean_performance"].clear()
        self.training_stats["recovery_points"].append(self.stage_episode)
        
        print(f"   Fresh start initiated - all archives preserved")
    
    def save_failure_learning_stats(self):
        """Save comprehensive failure learning statistics to disk"""
        try:
            import json
            
            def convert_to_json_serializable(obj):
                """Convert numpy types to Python native types for JSON serialization"""
                import numpy as np
                
                if isinstance(obj, (np.integer, np.floating, np.complexfloating)):
                    return obj.item()  # Convert numpy scalars
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()  # Convert numpy arrays
                elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):  # Any other numpy-like scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_json_serializable(item) for item in obj]
                elif isinstance(obj, deque):
                    return [convert_to_json_serializable(item) for item in obj]
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif obj is None or isinstance(obj, (str, int, float)):
                    return obj
                else:
                    # Fallback: try to convert to string for unknown types
                    try:
                        return str(obj)
                    except:
                        return f"<non-serializable: {type(obj).__name__}>"
            
            # Compile complete statistics
            stats_summary = {
                "session_summary": {
                    "total_episodes": self.episode,
                    "stage_episodes": self.stage_episode,
                    "current_stage": self.current_n_goals,
                    "failure_archives_count": len(self.failure_archives),
                    "recovery_attempts_count": len(self.recovery_attempts),
                    "training_episodes_since_reset": self.training_stats["episodes_since_reset"],
                    "session_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                
                "failure_learning_data": {
                    "failure_archives": self.failure_archives,
                    "recovery_attempts": self.recovery_attempts,
                    "risk_score_history": list(self.failure_risk_history),
                    "research_stats": {
                        "raw_performance": self.research_stats["raw_performance"],
                        "failure_episodes": self.research_stats["failure_episodes"],
                        "intervention_outcomes": self.research_stats["intervention_outcomes"],
                        "risk_scores": self.research_stats["risk_scores"]
                    }
                },
                
                "training_stats": {
                    "clean_performance": list(self.training_stats["clean_performance"]),
                    "recovery_points": self.training_stats["recovery_points"],
                    "episodes_since_reset": self.training_stats["episodes_since_reset"]
                }
            }
            
            # Convert all data to JSON-serializable format
            serializable_stats = convert_to_json_serializable(stats_summary)
            
            # Save main statistics file
            stats_file = f"{self.results_dir}/failure_learning_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(serializable_stats, f, indent=2)
                
            # Save compact summary for quick analysis
            summary_file = f"{self.results_dir}/failure_summary.json"
            compact_summary = {
                "session_id": f"stage_{self.current_n_goals}_ep_{self.episode}",
                "failure_count": len(self.failure_archives),
                "intervention_count": len(self.recovery_attempts),
                "current_risk": self.failure_risk_history[-1] if self.failure_risk_history else 0.0,
                "max_risk": max(self.failure_risk_history) if self.failure_risk_history else 0.0,
                "clean_episodes": len(self.training_stats["clean_performance"]),
                "raw_episodes": len(self.research_stats["raw_performance"]),
                "intervention_types": list(set(attempt["intervention"] for attempt in self.recovery_attempts)),
                "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Convert compact summary as well
            serializable_summary = convert_to_json_serializable(compact_summary)
            
            with open(summary_file, 'w') as f:
                json.dump(serializable_summary, f, indent=2)
                
            print(f"📊 Failure learning stats saved: {stats_file}")
            return True
            
        except Exception as e:
            print(f"Error saving failure learning stats: {e}")
            return False
    
