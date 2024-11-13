# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

from .PPOTransformer import PPOTransformer
from PoliwhiRL.models.ICM import ICMModule


class PPOModel:
    def __init__(self, input_shape, action_size, config):
        self.config = config
        self.input_shape = input_shape
        self.action_size = action_size
        self.device = torch.device(self.config["device"])

        # Base parameters
        self.base_learning_rate = self.config["ppo_learning_rate"]
        self.learning_rate = self.base_learning_rate  # Keep for compatibility
        self.gamma = self.config["ppo_gamma"]
        self.epsilon = self.config["ppo_epsilon"]
        self.value_loss_coef = self.config["ppo_value_loss_coef"]
        self.entropy_coef = self.config["ppo_entropy_coef"]
        self.entropy_decay = self.config["ppo_entropy_coef_decay"]
        self.entropy_min = self.config["ppo_entropy_coef_min"]

        # Reference values for adaptation
        self.reference_episode_length = 1000  # Base episode length
        self.reference_update_freq = 128      # Base update frequency
        self._last_episode_length = self.config["episode_length"]
        self._last_update_freq = self.config["ppo_update_frequency"]

        self._initialize_networks()
        self._initialize_optimizers()

    def _calculate_adaptive_learning_rate(self, episode_length, update_frequency):
        """
        Adjust learning rate based on episode length and update frequency.
        """
        length_factor = (self.reference_episode_length / episode_length) ** 0.5
        update_factor = (self.reference_update_freq / update_frequency) ** 0.5
        
        # Combine factors and clip to reasonable bounds
        adaptation_factor = min(max(length_factor * update_factor, 0.1), 10.0)
        return self.base_learning_rate * adaptation_factor

    def _initialize_networks(self):
        self.actor_critic = PPOTransformer(self.input_shape, self.action_size).to(
            self.device
        )
        self.icm = ICMModule(self.input_shape, self.action_size, self.config)

    def _initialize_optimizers(self):
        self.current_lr = self._calculate_adaptive_learning_rate(
            self.config["episode_length"],
            self.config["ppo_update_frequency"]
        )
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.current_lr
        )
        self._setup_lr_scheduler()

    def _setup_lr_scheduler(self):
        """
        Set up a cyclical learning rate scheduler with adaptive bounds
        """
        # Calculate cycle length based on episode length
        steps_per_episode = self.config["episode_length"] // self.config["ppo_update_frequency"]
        cycle_length = max(100, steps_per_episode // 2)
        
        self.scheduler = CyclicLR(
            self.optimizer,
            base_lr=self.current_lr * 0.1,  # Lower bound
            max_lr=self.current_lr,         # Upper bound
            step_size_up=cycle_length,
            mode="triangular2",
            cycle_momentum=False
        )

    def adapt_to_new_parameters(self, episode_length=None, update_frequency=None):
        """
        Adapt the model's learning parameters to new episode length or update frequency
        """
        if episode_length is not None:
            self.config["episode_length"] = episode_length
        
        if update_frequency is not None:
            self.config["ppo_update_frequency"] = update_frequency
        
        # Calculate new learning rate
        self.current_lr = self._calculate_adaptive_learning_rate(
            self.config["episode_length"],
            self.config["ppo_update_frequency"]
        )
        
        # Update optimizer with new learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        # Reset scheduler with new parameters
        self._setup_lr_scheduler()
        
        return self.current_lr

    def get_action(self, state_sequence):
        state_sequence = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_sequence)
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

    def compute_log_prob(self, state_sequence, action):
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        action_probs, _ = self.actor_critic(state_tensor)
        return torch.log(action_probs[0, action]).item()

    def compute_intrinsic_reward(self, state, next_state, action):
        return self.icm.compute_intrinsic_reward(state, next_state, action)

    def update(self, data, episode):
        # Check if parameters have changed and adapt if necessary
        current_episode_length = self.config["episode_length"]
        current_update_freq = self.config["ppo_update_frequency"]
        
        if (current_episode_length != self._last_episode_length or
            current_update_freq != self._last_update_freq):
            self.adapt_to_new_parameters(current_episode_length, current_update_freq)
            self._last_episode_length = current_episode_length
            self._last_update_freq = current_update_freq

        actor_loss, critic_loss, entropy_loss = self._compute_ppo_losses(data, episode)
        icm_loss = self.icm.update(
            data["states"][:, -1], data["next_states"][:, -1], data["actions"]
        )

        loss = actor_loss + critic_loss + entropy_loss
        self._update_networks(loss)

        return loss.item(), icm_loss

    def save(self, path):
        """
        Save model state and parameters securely
        """
        # Save network states
        torch.save(self.actor_critic.state_dict(), f"{path}/actor_critic.pth")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth", )
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler.pth",)
        
        # Save adaptive parameters separately in a structured format
        adaptive_params = {
            'current_lr': self.current_lr,
            'reference_episode_length': self.reference_episode_length,
            'reference_update_freq': self.reference_update_freq,
            'last_episode_length': self._last_episode_length,
            'last_update_freq': self._last_update_freq
        }
        
        # Convert to tensor for secure saving
        torch.save({k: torch.tensor(v) if isinstance(v, (int, float)) else v 
                    for k, v in adaptive_params.items()}, 
                f"{path}/adaptive_params.pth")
        
        self.icm.save(f"{path}/icm")

    def load(self, path):
        """
        Load model state and parameters securely
        """
        try:
            # Add safe globals for numpy
            torch.serialization.add_safe_globals(['numpy', 'np'])
            
            # Load network states
            self.actor_critic.load_state_dict(
                torch.load(f"{path}/actor_critic.pth", 
                        map_location=self.device, 
                        weights_only=True)
            )
            
            self.optimizer.load_state_dict(
                torch.load(f"{path}/optimizer.pth", 
                        map_location=self.device, 
                        weights_only=True)
            )
            
            self.scheduler.load_state_dict(
                torch.load(f"{path}/scheduler.pth", 
                        map_location=self.device, 
                        weights_only=True)
            )
            
            # Load adaptive parameters
            try:
                adaptive_params = torch.load(f"{path}/adaptive_params.pth", 
                                        map_location=self.device,
                                        weights_only=True)
                
                # Convert tensor values back to python types
                self.current_lr = adaptive_params['current_lr'].item()
                self.reference_episode_length = adaptive_params['reference_episode_length'].item()
                self.reference_update_freq = adaptive_params['reference_update_freq'].item()
                self._last_episode_length = adaptive_params['last_episode_length'].item()
                self._last_update_freq = adaptive_params['last_update_freq'].item()
                
                # Update optimizer with loaded learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                
                # Reset scheduler with loaded parameters
                self._setup_lr_scheduler()
                
            except FileNotFoundError:
                print("No adaptive parameters found, using defaults")
                self.current_lr = self.base_learning_rate
                self._last_episode_length = self.config["episode_length"]
                self._last_update_freq = self.config["ppo_update_frequency"]
            
            self.icm.load(f"{path}/icm")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with initial values")
            # Reset to initial state
            self.current_lr = self.base_learning_rate
            self._last_episode_length = self.config["episode_length"]
            self._last_update_freq = self.config["ppo_update_frequency"]
            
    def step_scheduler(self):
        self.scheduler.step()

    def _get_entropy_coef(self, episode):
        return max(self.entropy_coef * self.entropy_decay**episode, self.entropy_min)

    def _compute_ppo_losses(self, data, episode):
        returns = self._compute_returns(data["rewards"], data["dones"])
        advantages = self._compute_advantages(data["states"], returns)

        new_probs, new_values = self.actor_critic(data["states"])
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

    def _compute_advantages(self, states, returns):
        with torch.no_grad():
            _, state_values = self.actor_critic(states)
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
