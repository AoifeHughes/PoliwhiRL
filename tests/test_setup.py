import pytest
from unittest.mock import patch, MagicMock

from PoliwhiRL import Controller, RainbowDQN, PrioritizedReplayBuffer
from PoliwhiRL.models.RainbowDQN import setup_environment, initialize_training

@pytest.fixture
def mock_config():
    return {
        "use_grayscale": False,
        "device": "cpu",
        "learning_rate": 0.001,
        "capacity": 10000,
        "alpha": 0.6,
        "action_space": [0, 1],  # Simplified action space for testing
        "screen_size": (84, 84),
    }

@pytest.fixture
def mock_env(mock_config):
    env = MagicMock()
    env.screen_size.return_value = mock_config["screen_size"]
    env.action_space = mock_config["action_space"]
    return env

def test_controller_creation(mock_config):
    controller = setup_environment(mock_config)
    assert isinstance(controller, Controller), "Controller should be an instance of the Controller class"

def test_rainbow_dqn_creation(mock_config, mock_env):
    policy_net = RainbowDQN((1, *mock_env.screen_size()), len(mock_env.action_space), mock_config["device"])
    assert isinstance(policy_net, RainbowDQN), "policy_net should be an instance of RainbowDQN class"

def test_prioritized_replay_buffer_creation(mock_config):
    replay_buffer = PrioritizedReplayBuffer(mock_config["capacity"], mock_config["alpha"])
    assert isinstance(replay_buffer, PrioritizedReplayBuffer), "replay_buffer should be an instance of PrioritizedReplayBuffer class"

def test_initialize_training(mock_config, mock_env):
    with patch('your_project_path.load_checkpoint', return_value=None):
        policy_net, target_net, optimizer, replay_buffer = initialize_training(mock_config, mock_env)
        assert isinstance(policy_net, RainbowDQN), "policy_net should be an instance of RainbowDQN"
        assert isinstance(target_net, RainbowDQN), "target_net should be an instance of RainbowDQN"
        assert hasattr(optimizer, 'step'), "optimizer should have a 'step' method"
        assert isinstance(replay_buffer, PrioritizedReplayBuffer), "replay_buffer should be an instance of PrioritizedReplayBuffer"
