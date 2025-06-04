# -*- coding: utf-8 -*-
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PoliwhiRL.models.PPO.ppo_model_implementation import PPOModel


class TestGAE:
    """Test Generalized Advantage Estimation implementation"""

    @pytest.fixture
    def config(self):
        return {
            "device": "cpu",
            "ppo_learning_rate": 0.001,
            "ppo_gamma": 0.99,
            "ppo_lambda": 0.95,
            "ppo_epsilon": 0.2,
            "ppo_value_loss_coef": 0.5,
            "ppo_entropy_coef": 0.01,
            "ppo_entropy_coef_decay": 0.99,
            "ppo_entropy_coef_min": 0.001,
            "ppo_exploration_history_length": 5,
            "ppo_intrinsic_reward_scale": 0.01,
            "database_use": False,
            "icm_learning_rate": 0.001,
            "icm_beta": 0.2,
            "inverse_loss_scale": 0.2,
            "forward_loss_scale": 0.8,
            "icm_curiosity_weight": 0.1,
            "icm_loss_scale": 0.1
        }

    @pytest.fixture
    def ppo_model(self, config):
        input_shape = (84, 84, 3)
        action_size = 4
        return PPOModel(input_shape, action_size, config)

    def test_gae_calculation_simple(self, ppo_model):
        """Test GAE calculation with simple values"""
        # Create simple test data
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        values = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
        dones = torch.tensor([False, False, True], dtype=torch.bool)
        
        # Create dummy states for the actor-critic
        states = torch.randn(3, 10, 84, 84, 3)  # Assuming sequence length of 10
        
        # Mock the actor_critic to return a fixed last value
        with patch.object(ppo_model.actor_critic, 'forward') as mock_forward:
            mock_forward.return_value = (None, torch.tensor([[0.0]]))
            
            returns, advantages = ppo_model._compute_gae(
                rewards, values, dones, states
            )
        
        # Verify shapes
        assert returns.shape == rewards.shape
        assert advantages.shape == rewards.shape
        
        # Verify returns are reasonable
        assert torch.all(torch.isfinite(returns))
        assert torch.all(torch.isfinite(advantages))

    def test_gae_with_terminal_state(self, ppo_model):
        """Test GAE handling of terminal states"""
        rewards = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float32)
        values = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float32)
        dones = torch.tensor([False, False, True], dtype=torch.bool)
        states = torch.randn(3, 10, 84, 84, 3)
        
        with patch.object(ppo_model.actor_critic, 'forward') as mock_forward:
            mock_forward.return_value = (None, torch.tensor([[0.0]]))
            
            returns, advantages = ppo_model._compute_gae(
                rewards, values, dones, states
            )
        
        # Last advantage should consider terminal state
        # In terminal state, next value should be 0
        assert torch.all(torch.isfinite(advantages))

    def test_gae_vs_simple_returns(self, ppo_model):
        """Compare GAE returns with simple discounted returns"""
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        values = torch.tensor([2.0, 1.8, 1.5, 1.0], dtype=torch.float32)
        dones = torch.tensor([False, False, False, False], dtype=torch.bool)
        states = torch.randn(4, 10, 84, 84, 3)
        
        # Compute GAE
        with patch.object(ppo_model.actor_critic, 'forward') as mock_forward:
            mock_forward.return_value = (None, torch.tensor([[0.5]]))
            
            gae_returns, gae_advantages = ppo_model._compute_gae(
                rewards, values, dones, states
            )
        
        # Compute simple returns for comparison
        simple_returns = ppo_model._compute_returns(rewards, dones)
        
        # GAE returns should be different from simple returns due to value bootstrapping
        assert not torch.allclose(gae_returns, simple_returns, atol=0.01)

    def test_gae_lambda_effect(self, ppo_model):
        """Test that lambda parameter affects GAE calculation"""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        values = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
        dones = torch.tensor([False, False, False, False], dtype=torch.bool)
        states = torch.randn(4, 10, 84, 84, 3)
        
        # Calculate with default lambda (0.95)
        with patch.object(ppo_model.actor_critic, 'forward') as mock_forward:
            mock_forward.return_value = (None, torch.tensor([[4.0]]))
            _, advantages_high_lambda = ppo_model._compute_gae(
                rewards, values, dones, states
            )
        
        # Calculate with lower lambda
        ppo_model.lambda_ = 0.5
        with patch.object(ppo_model.actor_critic, 'forward') as mock_forward:
            mock_forward.return_value = (None, torch.tensor([[4.0]]))
            _, advantages_low_lambda = ppo_model._compute_gae(
                rewards, values, dones, states
            )
        
        # Advantages should be different with different lambda values
        assert not torch.allclose(advantages_high_lambda, advantages_low_lambda)

    def test_gae_normalization(self, ppo_model):
        """Test that advantages are properly normalized"""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        dones = torch.tensor([False, False, False, False, False], dtype=torch.bool)
        states = torch.randn(5, 10, 84, 84, 3)
        
        with patch.object(ppo_model.actor_critic, 'forward') as mock_forward:
            mock_forward.return_value = (None, torch.tensor([[5.0]]))
            _, advantages = ppo_model._compute_gae(
                rewards, values, dones, states
            )
        
        # Check normalization (mean should be close to 0, std close to 1)
        assert abs(advantages.mean().item()) < 0.1
        assert abs(advantages.std().item() - 1.0) < 0.2

    def test_gae_single_step(self, ppo_model):
        """Test GAE with single step (edge case)"""
        rewards = torch.tensor([1.0], dtype=torch.float32)
        values = torch.tensor([0.5], dtype=torch.float32)
        dones = torch.tensor([True], dtype=torch.bool)
        states = torch.randn(1, 10, 84, 84, 3)
        
        with patch.object(ppo_model.actor_critic, 'forward') as mock_forward:
            mock_forward.return_value = (None, torch.tensor([[0.0]]))
            returns, advantages = ppo_model._compute_gae(
                rewards, values, dones, states
            )
        
        # Should handle single step without errors
        assert returns.shape == torch.Size([1])
        assert advantages.shape == torch.Size([1])
        assert torch.all(torch.isfinite(returns))
        assert torch.all(torch.isfinite(advantages))

    def test_ppo_losses_use_gae(self, ppo_model):
        """Test that PPO losses properly use GAE when values are available"""
        # Create test data with values
        data = {
            "states": torch.randn(5, 10, 84, 84, 3),
            "actions": torch.randint(0, 4, (5,)),
            "rewards": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            "dones": torch.tensor([False, False, False, False, True]),
            "old_log_probs": torch.randn(5),
            "values": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        }
        
        # Spy on _compute_gae to ensure it's called
        with patch.object(ppo_model, '_compute_gae', wraps=ppo_model._compute_gae) as spy_gae:
            actor_loss, critic_loss, entropy_loss = ppo_model._compute_ppo_losses(
                data, episode=0
            )
            
            # Verify GAE was called
            spy_gae.assert_called_once()
            
        # Verify losses are computed
        assert torch.isfinite(actor_loss)
        assert torch.isfinite(critic_loss)
        assert torch.isfinite(entropy_loss)

    def test_ppo_losses_fallback_without_values(self, ppo_model):
        """Test that PPO falls back to simple advantages when values are not available"""
        # Create test data without values
        data = {
            "states": torch.randn(5, 10, 84, 84, 3),
            "actions": torch.randint(0, 4, (5,)),
            "rewards": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            "dones": torch.tensor([False, False, False, False, True]),
            "old_log_probs": torch.randn(5)
        }
        
        # Spy on both methods
        with patch.object(ppo_model, '_compute_gae', wraps=ppo_model._compute_gae) as spy_gae:
            with patch.object(ppo_model, '_compute_returns', wraps=ppo_model._compute_returns) as spy_returns:
                actor_loss, critic_loss, entropy_loss = ppo_model._compute_ppo_losses(
                    data, episode=0
                )
                
                # Verify GAE was not called but simple returns were
                spy_gae.assert_not_called()
                spy_returns.assert_called_once()
        
        # Verify losses are computed
        assert torch.isfinite(actor_loss)
        assert torch.isfinite(critic_loss)
        assert torch.isfinite(entropy_loss)