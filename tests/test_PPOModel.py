# -*- coding: utf-8 -*-
import os
import torch
from PoliwhiRL.models.PPO import PPOModel
from PoliwhiRL.models.PPO.training_functions import (
    compute_returns,
    load_latest_checkpoint,
    save_checkpoint,
    setup_optimizer,
    setup_scheduler,
)


def test_ppo_model():
    input_dim = (3, 84, 84)  # Example input dimension for vision
    num_actions = 4
    model = PPOModel(input_dim, num_actions, vision=True)

    # Test forward pass
    batch_size = 2
    seq_len = 5
    x = torch.rand(batch_size, seq_len, *input_dim)
    action_probs, value_estimates, hidden_state = model(x)

    assert action_probs.shape == (batch_size, seq_len, num_actions)
    assert value_estimates.shape == (batch_size, seq_len, 1)
    assert isinstance(hidden_state, tuple)
    assert len(hidden_state) == 2
    assert hidden_state[0].shape == (1, batch_size, model.lstm.hidden_size)
    assert hidden_state[1].shape == (1, batch_size, model.lstm.hidden_size)


def test_optimizer_and_scheduler():
    model = PPOModel((3, 84, 84), 4, vision=True)
    config = {"learning_rate": 1e-3}

    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 50
    assert scheduler.gamma == 0.9


def test_compute_returns():
    next_value = torch.tensor(0.5)
    rewards = [1.0, 2.0, 3.0]
    masks = [1.0, 1.0, 0.0]
    gamma = 0.99

    returns = compute_returns(next_value, rewards, masks, gamma)

    expected_returns = torch.tensor([5.9405, 4.9500, 3.0000])
    assert torch.allclose(returns, expected_returns, atol=1e-4)


def test_checkpoint_functions(tmp_path):
    model = PPOModel((3, 84, 84), 4, vision=True)
    checkpoint_dir = tmp_path / "checkpoints"
    os.makedirs(checkpoint_dir)

    # Test save_checkpoint
    save_checkpoint(model, checkpoint_dir, 100)
    assert os.path.exists(checkpoint_dir / "ppo_model_ep100.pth")

    # Test load_latest_checkpoint
    loaded_episode = load_latest_checkpoint(model, checkpoint_dir)
    assert loaded_episode == 100

    # Save another checkpoint and test if it loads the latest
    save_checkpoint(model, checkpoint_dir, 200)
    loaded_episode = load_latest_checkpoint(model, checkpoint_dir)
    assert loaded_episode == 200
