# -*- coding: utf-8 -*-
import torch
import torch.multiprocessing as mp
import json
import os
from collections import deque, defaultdict
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent
from tqdm import tqdm


class MultiAgentPPO:
    def __init__(self, config):
        self.config = config
        self.num_agents = config["ppo_num_agents"]
        self.iterations = config["ppo_iterations"]
        self.total_episodes_run = config["start_episode"]
        self.agents = {}
        self.agent_metrics = defaultdict(dict)

        # Initialize metrics storage for each agent
        for i in range(self.num_agents):
            self.agent_metrics[i] = self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize empty metrics structure with proper deque sizes"""
        return {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_losses": [],
            "episode_icm_losses": [],
            "moving_avg_reward": deque(maxlen=100),
            "moving_avg_length": deque(maxlen=100),
            "moving_avg_loss": deque(maxlen=100),
            "moving_avg_icm_loss": deque(maxlen=100),
            "buttons_pressed": deque(maxlen=100),
            "current_episode": 0,  # Add episode counter to metrics
        }

    def average_weights(self, agent_paths, input_shape, action_size):
        """Average the weights of all agents while preserving metrics and episode counts"""
        averaged_agent = PPOAgent(input_shape, action_size, self.config)

        actor_critic_params = []
        icm_params = []
        optimizer_states = []
        scheduler_states = []
        num_episodes = {}
        max_episode_count = 0
        agent_count = 0

        # Collect parameters and track max episode count
        for path in agent_paths:
            try:
                agent = PPOAgent(input_shape, action_size, self.config)
                agent.load_model(path)
                # Load metrics to get episode count
                agent_id = int(path.split("_")[-1])
                metrics = self.load_agent_metrics(agent_id)
                max_episode_count = max(
                    max_episode_count, metrics.get("current_episode", 0)
                )

                agent_count += 1

                actor_critic_params.append(
                    {
                        name: param.data.clone()
                        for name, param in agent.model.actor_critic.named_parameters()
                    }
                )
                icm_params.append(
                    {
                        name: param.data.clone()
                        for name, param in agent.model.icm.icm.named_parameters()
                    }
                )
                optimizer_states.append(agent.model.optimizer.state_dict())
                scheduler_states.append(agent.model.scheduler.state_dict())
                num_episodes[path] = len(metrics["episode_rewards"])
            except Exception as e:
                print(f"Error loading agent from {path}: {e}")
                continue

        if not actor_critic_params:
            raise ValueError("No valid agents found to average")

        # Average parameters while preserving episode count
        self._average_parameters(
            averaged_agent,
            actor_critic_params,
            icm_params,
            optimizer_states,
            scheduler_states,
            agent_count,
        )

        # Set the episode count to the maximum across all agents
        metrics = self._initialize_metrics()
        metrics["current_episode"] = max_episode_count
        averaged_agent.set_episode_data(metrics)

        return averaged_agent, max_episode_count, num_episodes

    def _average_parameters(
        self,
        averaged_agent,
        actor_critic_params,
        icm_params,
        optimizer_states,
        scheduler_states,
        agent_count,
    ):
        """Helper method to average model parameters"""
        # Average actor_critic parameters
        for name in actor_critic_params[0].keys():
            averaged_param = (
                sum(params[name] for params in actor_critic_params) / agent_count
            )
            averaged_agent.model.actor_critic.get_parameter(name).data.copy_(
                averaged_param
            )

        # Average ICM parameters
        for name in icm_params[0].keys():
            averaged_param = sum(params[name] for params in icm_params) / agent_count
            averaged_agent.model.icm.icm.get_parameter(name).data.copy_(averaged_param)

        # Average optimizer state
        averaged_optimizer_state = optimizer_states[0].copy()
        for key in averaged_optimizer_state["state"].keys():
            for param_key in averaged_optimizer_state["state"][key].keys():
                if isinstance(
                    averaged_optimizer_state["state"][key][param_key], torch.Tensor
                ):
                    averaged_optimizer_state["state"][key][param_key] = (
                        sum(
                            state["state"][key][param_key] for state in optimizer_states
                        )
                        / agent_count
                    )
        averaged_agent.model.optimizer.load_state_dict(averaged_optimizer_state)

        # Average scheduler state
        averaged_scheduler_state = scheduler_states[0].copy()
        for key, value in averaged_scheduler_state.items():
            if isinstance(value, torch.Tensor):
                averaged_scheduler_state[key] = (
                    sum(state[key] for state in scheduler_states) / agent_count
                )
        averaged_agent.model.scheduler.load_state_dict(averaged_scheduler_state)

    def save_agent_metrics(self, agent_id, metrics, current_episode=None):
        """Save metrics for each agent separately using JSON"""
        metrics_dir = f"{self.config['results_dir']}/agent_{agent_id}"
        os.makedirs(metrics_dir, exist_ok=True)

        # Update current episode if provided
        if current_episode is not None:
            metrics["current_episode"] = current_episode

        # Update stored metrics
        self.agent_metrics[agent_id] = {
            "episode_rewards": metrics["episode_rewards"],
            "episode_lengths": metrics["episode_lengths"],
            "episode_losses": metrics["episode_losses"],
            "episode_icm_losses": metrics["episode_icm_losses"],
            "moving_avg_reward": deque(metrics["moving_avg_reward"], maxlen=100),
            "moving_avg_length": deque(metrics["moving_avg_length"], maxlen=100),
            "moving_avg_loss": deque(metrics["moving_avg_loss"], maxlen=100),
            "moving_avg_icm_loss": deque(metrics["moving_avg_icm_loss"], maxlen=100),
            "buttons_pressed": deque(metrics["buttons_pressed"], maxlen=100),
            "current_episode": metrics.get("current_episode", 0),
        }

        # Convert deques to lists for JSON serialization
        json_metrics = {
            "episode_rewards": metrics["episode_rewards"],
            "episode_lengths": metrics["episode_lengths"],
            "episode_losses": metrics["episode_losses"],
            "episode_icm_losses": metrics["episode_icm_losses"],
            "moving_avg_reward": list(metrics["moving_avg_reward"]),
            "moving_avg_length": list(metrics["moving_avg_length"]),
            "moving_avg_loss": list(metrics["moving_avg_loss"]),
            "moving_avg_icm_loss": list(metrics["moving_avg_icm_loss"]),
            "buttons_pressed": list(metrics["buttons_pressed"]),
            "current_episode": metrics.get("current_episode", 0),
            "total_episodes": len(metrics["episode_rewards"]),
        }
        for data in all_episode_data:
            for key in combined_data:
                if isinstance(data[key], list):
                    combined_data[key].extend(data[key][len(combined_data[key]) :])
                else:
                    combined_data[key] += data[key]

        with open(f"{metrics_dir}/metrics.json", "w") as f:
            json.dump(json_metrics, f)

    def load_agent_metrics(self, agent_id):
        """Load metrics for a specific agent"""
        metrics_path = f"{self.config['results_dir']}/agent_{agent_id}/metrics.json"
        try:
            with open(metrics_path, "r") as f:
                json_metrics = json.load(f)

            # Restore metrics with proper deque initialization
            metrics = {
                "episode_rewards": json_metrics["episode_rewards"],
                "episode_lengths": json_metrics["episode_lengths"],
                "episode_losses": json_metrics["episode_losses"],
                "episode_icm_losses": json_metrics["episode_icm_losses"],
                "moving_avg_reward": deque(
                    json_metrics["moving_avg_reward"], maxlen=100
                ),
                "moving_avg_length": deque(
                    json_metrics["moving_avg_length"], maxlen=100
                ),
                "moving_avg_loss": deque(json_metrics["moving_avg_loss"], maxlen=100),
                "moving_avg_icm_loss": deque(
                    json_metrics["moving_avg_icm_loss"], maxlen=100
                ),
                "buttons_pressed": deque(json_metrics["buttons_pressed"], maxlen=100),
                "current_episode": json_metrics.get("current_episode", 0),
            }

            # Update stored metrics
            self.agent_metrics[agent_id] = metrics

            # Update total episodes if available
            if "total_episodes" in json_metrics:
                self.total_episodes_run = max(
                    self.total_episodes_run, json_metrics["total_episodes"]
                )

            return metrics
        except Exception as e:
            print(f"Error loading metrics for agent {agent_id}: {e}")
            return self._initialize_metrics()

    def run_agent(self, i, state_shape, num_actions, checkpoint_path, start_episode):
        """Run training for a single agent with proper episode tracking"""
        try:
            config = self.config.copy()
            config["checkpoint"] = f"{checkpoint_path}/agent_{i}"
            config["record_path"] = f"{self.config['record_path']}/agent_{i}"
            config["export_state_loc"] = f"{self.config['export_state_loc']}/agent_{i}"
            config["results_dir"] = f"{self.config['results_dir']}/agent_{i}"
            agent = PPOAgent(state_shape, num_actions, config)

            # Load existing metrics and update episode count
            existing_metrics = self.load_agent_metrics(i)
            existing_metrics["current_episode"] = start_episode
            agent.set_episode_data(existing_metrics)

            agent.load_model(config["checkpoint"])
            # agent.episode = start_episode

            if config["use_curriculum"]:
                agent.run_curriculum(1, config["N_goals_target"], 600)
            else:
                agent.train_agent()

            # Update and save metrics with current episode count
            metrics = agent.get_episode_data()
            metrics["current_episode"] = start_episode + self.config["num_episodes"]
            self.save_agent_metrics(i, metrics)
            agent.save_model(config["checkpoint"])

        except Exception as e:
            print(f"Error in agent {i}: {e}")
            return None

    def distribute_averaged_weights(self, averaged_agent, agent_paths, num_episodes):
        """Distribute averaged weights to all agents while preserving metrics and episode count"""
        for i, path in enumerate(agent_paths):
            try:
                averaged_agent.episode = num_episodes[path]
                # Load existing metrics before saving
                metrics = self.agent_metrics[i]
                # Update the averaged agent's metrics with the preserved ones
                averaged_agent.set_episode_data(metrics)

                # Save the agent with updated metrics
                averaged_agent.save_model(path)
                self.save_agent_metrics(i, metrics, averaged_agent.episode)
            except Exception as e:
                print(f"Error saving averaged weights to {path}: {e}")

    def train(self):
        """Train multiple agents in parallel with proper episode tracking"""
        env = Env(self.config)
        state_shape = env.output_shape()
        num_actions = env.action_space.n
        checkpoint_path = self.config["checkpoint"]

        for iteration in tqdm(range(self.iterations), desc="Training Iterations"):
            # Run agents in parallel with current episode count
            with mp.Pool(processes=self.num_agents) as pool:
                pool.starmap(
                    self.run_agent,
                    [
                        (
                            i,
                            state_shape,
                            num_actions,
                            checkpoint_path,
                            self.total_episodes_run,
                        )
                        for i in range(self.num_agents)
                    ],
                )

            # Average weights and distribute back to agents
            agent_paths = [
                f"{checkpoint_path}/agent_{i}" for i in range(self.num_agents)
            ]
            averaged_agent, max_episode_count, num_episodes = self.average_weights(
                agent_paths, state_shape, num_actions
            )

            # Update total episodes run
            self.total_episodes_run = max_episode_count + self.config["num_episodes"]

            # Distribute averaged weights while preserving episode count
            self.distribute_averaged_weights(averaged_agent, agent_paths, num_episodes)

            # Print progress if needed
            # self.print_agent_progress(iteration)

        return averaged_agent.model

    def print_agent_progress(self, iteration):
        """Print training progress for each agent"""
        print(f"\nIteration {iteration + 1}/{self.iterations} Summary:")
        for i in range(self.num_agents):
            metrics = self.load_agent_metrics(i)
            if metrics and metrics["episode_rewards"]:
                recent_rewards = metrics["episode_rewards"][-100:]
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                max_reward = max(recent_rewards)
                print(f"Agent {i}:")
                print(f"  Average Reward (last 100 episodes): {avg_reward:.2f}")
                print(f"  Max Reward (last 100 episodes): {max_reward:.2f}")
                print(f"  Total Episodes: {len(metrics['episode_rewards'])}")
                print(f"  Current Episode: {metrics['current_episode']}")

    def get_agent_metrics(self, agent_id):
        """Get metrics for a specific agent"""
        return self.load_agent_metrics(agent_id)

    def get_all_agent_metrics(self):
        """Get metrics for all agents"""
        return {i: self.load_agent_metrics(i) for i in range(self.num_agents)}
