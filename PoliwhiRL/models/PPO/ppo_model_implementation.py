# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

from .PPOTransformer import PPOTransformer
from PoliwhiRL.models.ICM import ICMModule


class PPOModel:
    def __init__(self, input_shape, action_size, config):
        self.config = config
        self.input_shape = input_shape
        self.action_size = action_size
        self.device = torch.device(self.config["device"])

        self.learning_rate = self.config["ppo_learning_rate"]
        self.gamma = self.config["ppo_gamma"]
        self.lambda_ = self.config.get("ppo_lambda", 0.95)
        self.epsilon = self.config["ppo_epsilon"]
        self.value_loss_coef = self.config["ppo_value_loss_coef"]
        self.entropy_coef = self.config["ppo_entropy_coef"]
        self.entropy_decay = self.config["ppo_entropy_coef_decay"]
        self.entropy_min = self.config["ppo_entropy_coef_min"]
        self.entropy_decay_mode = self.config.get("ppo_entropy_decay_mode", "episode")
        
        # For step-based decay tracking
        self.total_steps = 0
        self.step_decay_interval = self.config.get("ppo_step_decay_interval", 1000)

        self._initialize_networks()
        
        # Initialize regularization for catastrophic forgetting prevention
        self.regularization_weight = self.config.get("regularization_weight", 0.01)
        self.reference_params = None  # Will store reference parameters
        self.update_reference_frequency = self.config.get("update_reference_frequency", 50)
        self.update_count = 0
        self._initialize_optimizers()
        
        # Set initial reference parameters
        self._update_reference_params()

    def _initialize_networks(self):
        # Pass ppo_exploration_history_length from config if available
        ppo_exploration_history_length = self.config.get(
            "ppo_exploration_history_length", 5
        )
        self.actor_critic = PPOTransformer(
            self.input_shape,
            self.action_size,
            ppo_exploration_history_length=ppo_exploration_history_length,
        ).to(self.device)
        self.icm = ICMModule(self.input_shape, self.action_size, self.config)

    def _initialize_optimizers(self):
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.learning_rate
        )
        self._setup_lr_scheduler()

    def _setup_lr_scheduler(self):
        # Use an even gentler exponential decay
        # 0.9997 means LR will be ~97% after 100 episodes, ~91% after 300 episodes
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.9997  # Much gentler decay than 0.999
        )

    def get_action(self, state_sequence, exploration_tensor=None, game_state=None):
        state_sequence = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        if exploration_tensor is not None:
            exploration_tensor = (
                torch.FloatTensor(exploration_tensor).unsqueeze(0).to(self.device)
            )
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_sequence, exploration_tensor, game_state=game_state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def get_action_half_precision(self, state_sequence):
        with torch.no_grad():
            state_sequence = (
                torch.HalfTensor(state_sequence).unsqueeze(0).to(self.device)
            )
            action_probs, _ = self.actor_critic.half()(state_sequence)
        action = torch.multinomial(action_probs.float(), 1).item()
        return action

    def compute_log_prob(self, state_sequence, action, exploration_tensor=None, game_state=None):
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        if exploration_tensor is not None:
            exploration_tensor = (
                torch.FloatTensor(exploration_tensor).unsqueeze(0).to(self.device)
            )
        action_probs, _ = self.actor_critic(state_tensor, exploration_tensor, game_state=game_state)
        return torch.log(action_probs[0, action]).item()

    def compute_intrinsic_reward(self, state, next_state, action):
        return self.icm.compute_intrinsic_reward(state, next_state, action)

    def update(self, data, episode):
        # Track total steps for step-based entropy decay
        if self.entropy_decay_mode == "step":
            self.total_steps += len(data["states"])
        
        actor_loss, critic_loss, entropy_loss = self._compute_ppo_losses(data, episode)
        icm_loss = self.icm.update(
            data["states"][:, -1], data["next_states"][:, -1], data["actions"]
        )

        # Add regularization to prevent catastrophic forgetting
        regularization_loss = self._compute_regularization_loss()
        
        loss = actor_loss + critic_loss + entropy_loss + regularization_loss
        self._update_networks(loss)

        return loss.item(), icm_loss

    def _get_entropy_coef(self, episode):
        # Support both episode-based and step-based entropy decay
        if self.entropy_decay_mode == "step":
            return self._get_entropy_coef_step_based()
        else:
            return self._get_entropy_coef_episode_based(episode)
    
    def _get_entropy_coef_episode_based(self, episode):
        # CURRICULUM FIX: Use stage-relative episode count for entropy decay
        # This prevents entropy collapse across curriculum stages
        stage_episode = getattr(self, "stage_episode", episode)
        
        # MUCH gentler entropy decay - only after initial exploration phase
        if stage_episode < 50:
            # Keep high entropy for first 50 episodes of each stage
            base_entropy = self.entropy_coef
        else:
            # Very gentle decay after that - use 0.9995 instead of 0.999
            decay_episodes = stage_episode - 50
            base_entropy = max(
                self.entropy_coef * (self.entropy_decay ** decay_episodes), self.entropy_min
            )

        # Adaptive entropy boost for stage transitions
        entropy_boost = getattr(self, "entropy_boost", 0.0)
        
        # Combine base and boost
        final_entropy = base_entropy + entropy_boost
        
        # MUCH higher floor to prevent collapse - never below 60% of initial
        exploration_floor = max(self.entropy_coef * 0.6, 0.03)
        final_entropy = max(final_entropy, exploration_floor)
        
        # Cap at 2x initial to prevent instability
        max_entropy = self.entropy_coef * 2.0
        return min(final_entropy, max_entropy)
    
    def _get_entropy_coef_step_based(self):
        # Step-based entropy decay for long episodes
        decay_steps = self.total_steps // self.step_decay_interval
        
        # Keep high entropy for initial steps
        if self.total_steps < 5000:  # First 5k steps
            base_entropy = self.entropy_coef
        else:
            # Very gentle step-based decay
            base_entropy = max(
                self.entropy_coef * (self.entropy_decay ** decay_steps), self.entropy_min
            )
        
        # Adaptive entropy boost for exploration
        entropy_boost = getattr(self, "entropy_boost", 0.0)
        
        # Add periodic entropy boosts every 10k steps to prevent collapse
        if self.total_steps > 0 and self.total_steps % 10000 == 0:
            entropy_boost += 0.02  # Small periodic boost
        
        # Combine base and boost
        final_entropy = base_entropy + entropy_boost
        
        # Higher floor for long episodes - never below 70% of initial
        exploration_floor = max(self.entropy_coef * 0.7, 0.04)
        final_entropy = max(final_entropy, exploration_floor)
        
        # Cap at 2x initial to prevent instability
        max_entropy = self.entropy_coef * 2.0
        return min(final_entropy, max_entropy)

    def _compute_ppo_losses(self, data, episode):
        # Use GAE if we have values, otherwise fall back to simple advantages
        if "values" in data:
            returns, advantages = self._compute_gae(
                data["rewards"], data["values"], data["dones"], data["states"], 
                data.get("exploration_tensors"), data.get("game_states")
            )
        else:
            returns = self._compute_returns(data["rewards"], data["dones"])
            advantages = self._compute_advantages(
                data["states"], returns, data.get("exploration_tensors"), data.get("game_states")
            )

        new_probs, new_values = self.actor_critic(
            data["states"], data.get("exploration_tensors"), game_state=data.get("game_states")
        )
        new_probs = torch.clamp(new_probs, 1e-10, 1.0)
        new_log_probs = torch.log(
            new_probs.gather(1, data["actions"].unsqueeze(1)) + 1e-10
        ).squeeze()

        ratio = torch.exp(new_log_probs - data["old_log_probs"])
        ratio = torch.clamp(ratio, 0.0, 10.0)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        new_values = new_values.squeeze()
        if new_values.dim() == 0:
            new_values = new_values.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)

        critic_loss = self.value_loss_coef * nn.functional.mse_loss(new_values, returns)

        entropy = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=-1).mean()

        # Simpler entropy loss without penalty
        entropy_loss = -self._get_entropy_coef(episode) * entropy

        # Detailed debugging information if nans
        if (
            torch.isnan(actor_loss)
            or torch.isnan(critic_loss)
            or torch.isnan(entropy_loss)
        ):
            print(
                f"New probs range: ({new_probs.min().item()}, {new_probs.max().item()})"
            )
            print(
                f"New log probs range: ({new_log_probs.min().item()}, {new_log_probs.max().item()})"
            )
            print(f"Ratio range: ({ratio.min().item()}, {ratio.max().item()})")
            print(
                f"Advantages range: ({advantages.min().item()}, {advantages.max().item()})"
            )
            print(f"Returns range: ({returns.min().item()}, {returns.max().item()})")
            print(
                f"New values range: ({new_values.min().item()}, {new_values.max().item()})"
            )

        # Check for NaN or inf values
        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            print("Warning: NaN or inf detected in actor loss")
        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            print("Warning: NaN or inf detected in critic loss")
        if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
            print("Warning: NaN or inf detected in entropy loss")

        return actor_loss, critic_loss, entropy_loss

    def _update_networks(self, ppo_loss):
        self.optimizer.zero_grad()
        ppo_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
        self.optimizer.step()

    def _compute_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (~dones[t])
            returns[t] = running_return
        return returns

    def _compute_advantages(self, states, returns, exploration_tensors=None, game_states=None):
        with torch.no_grad():
            _, state_values = self.actor_critic(states, exploration_tensors, game_state=game_states)
            advantages = returns - state_values.squeeze()

            # Check for NaN values
            if torch.isnan(advantages).any():
                print("NaN detected in advantages before normalization")
                print(
                    f"Returns shape: {returns.shape}, range: ({returns.min().item()}, {returns.max().item()})"
                )
                print(
                    f"State values shape: {state_values.shape}, range: ({state_values.min().item()}, {state_values.max().item()})"
                )

            # Handle single state case
            if advantages.shape[0] > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            else:
                # For single state, we can't normalize, so we'll just use the raw advantage
                advantages = advantages - advantages.mean()

            # Check for NaN values after normalization
            if torch.isnan(advantages).any():
                print("NaN detected in advantages after normalization")
                advantages = torch.nan_to_num(advantages, nan=0.0)

        return advantages
    
    def _compute_gae(self, rewards, values, dones, states, exploration_tensors=None, game_states=None):
        """Compute Generalized Advantage Estimation (GAE)"""
        with torch.no_grad():
            # Get next state values
            # For the last state, we need to compute its value
            last_game_state = game_states[-1:] if game_states is not None else None
            _, last_value = self.actor_critic(
                states[-1:], exploration_tensors[-1:] if exploration_tensors is not None else None,
                game_state=last_game_state
            )
            
            # Compute returns and advantages using GAE
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)
            
            # Initialize GAE
            gae = 0
            next_value = last_value.squeeze()
            
            # Backward iteration through the trajectory
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t].float()
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t].float()
                    next_values = values[t + 1]
                
                # TD residual: r + γV(s') - V(s)
                delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
                
                # GAE: A_t = δ_t + γλA_{t+1}
                gae = delta + self.gamma * self.lambda_ * next_non_terminal * gae
                advantages[t] = gae
                
                # Returns for value function training
                returns[t] = advantages[t] + values[t]
            
            # Normalize advantages
            if advantages.shape[0] > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                advantages = advantages - advantages.mean()
                
        return returns, advantages

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), f"{path}/actor_critic.pth")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler.pth")
        
        # Save step tracking for step-based entropy decay
        model_info = {
            "total_steps": self.total_steps,
            "entropy_decay_mode": self.entropy_decay_mode
        }
        torch.save(model_info, f"{path}/model_info.pth")
        
        self.icm.save(f"{path}/icm")

    def load(self, path):
        self.actor_critic.load_state_dict(
            torch.load(
                f"{path}/actor_critic.pth", map_location=self.device, weights_only=True
            )
        )
        self.optimizer.load_state_dict(
            torch.load(
                f"{path}/optimizer.pth", map_location=self.device, weights_only=True
            )
        )
        self.scheduler.load_state_dict(
            torch.load(
                f"{path}/scheduler.pth", map_location=self.device, weights_only=True
            )
        )
        
        # Load step tracking for step-based entropy decay
        try:
            model_info = torch.load(
                f"{path}/model_info.pth", map_location=self.device, weights_only=True
            )
            self.total_steps = model_info.get("total_steps", 0)
            # Don't override entropy_decay_mode from config, just restore steps
        except FileNotFoundError:
            # Backwards compatibility - older checkpoints won't have this
            self.total_steps = 0
        
        self.icm.load(f"{path}/icm")

    def step_scheduler(self):
        self.scheduler.step()
    
    def reset_learning_rate_for_stage(self, stage_difficulty_multiplier=1.0):
        """Reset and boost learning rate for new curriculum stage"""
        # Calculate stage-appropriate learning rate
        base_lr = self.config["ppo_learning_rate"]
        # Reduce the boost multiplier for later stages
        boost_factor = max(0.3, 1.0 - stage_difficulty_multiplier * 0.1)
        boosted_lr = base_lr * boost_factor
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = boosted_lr
        
        # Reset scheduler to prevent accumulated decay
        self._setup_lr_scheduler()
        
        print(f"🔄 Learning rate reset to {boosted_lr:.6f} for new curriculum stage (stage {stage_difficulty_multiplier + 1})")
    
    def _compute_regularization_loss(self):
        """Compute regularization loss to prevent catastrophic forgetting."""
        if self.reference_params is None:
            return 0.0
        
        reg_loss = 0.0
        current_params = list(self.actor_critic.parameters())
        
        for current_param, reference_param in zip(current_params, self.reference_params):
            reg_loss += torch.sum((current_param - reference_param) ** 2)
        
        # Update reference parameters periodically
        self.update_count += 1
        if self.update_count % self.update_reference_frequency == 0:
            self._update_reference_params()
        
        return self.regularization_weight * reg_loss
    
    def _update_reference_params(self):
        """Update reference parameters for regularization."""
        self.reference_params = []
        for param in self.actor_critic.parameters():
            self.reference_params.append(param.clone().detach())
        
        if hasattr(self, 'update_count'):
            print(f"📌 Updated reference parameters at update {self.update_count}")
