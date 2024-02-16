# -*- coding: utf-8 -*-
import pytest
from PoliwhiRL.environment import Controller
from os.path import basename
import tempfile
import os


# Sample configuration to be returned by the mock
mock_config = {
    "rom_path": "./emu_files/Pokemon - Crystal Version.gbc",
    "state_path": "./emu_files/states/start.state",
    "episode_length": 5,
    "device": "cpu",
    "num_episodes": 5,
    "batch_size": 32,
    "checkpoint": "./checkpoints/RainbowDQN.pth",
    "model": "RainbowDQN",
    "sight": False,
    "erase": False,
    "parallel": False,
    "runs_per_worker": 4,
    "num_workers": 6,
    "checkpoint_interval": 100,
    "epsilon_by_location": False,
    "extra_files": [
        "./emu_files/Pokemon - Crystal Version.gbc.ram",
        "./emu_files/Pokemon - Crystal Version.gbc.rtc",
    ],
}


@pytest.fixture
def temp_state_file():
    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(suffix=".state")
    os.close(fd)  # Close the file descriptor

    yield temp_path  # Provide the path to the test

    # Cleanup
    os.remove(temp_path)


@pytest.fixture
def controller():
    rom_path = mock_config["rom_path"]
    state_path = mock_config["state_path"]
    episode_length = 25
    sight = False
    return Controller(
        rom_path,
        state_path,
        timeout=episode_length,
        log_path="./logs/rainbow_env_eval.json",
        use_sight=sight,
    )


def test_initialization_with_mocked_config(controller):
    # Test if the Controller uses values from the mocked config
    assert basename(controller.rom_path) == basename(mock_config["rom_path"])
    assert basename(controller.state_path) == basename(mock_config["state_path"])


def test_step_function_updates_state_correctly(controller):
    initial_steps = controller.steps
    controller.step(0)  # Assuming 0 corresponds to a valid action
    assert controller.steps == initial_steps + 1


def test_saving_and_loading_state(controller, temp_state_file):
    controller.store_controller_state(temp_state_file)
    controller.load_stored_controller_state(temp_state_file)


# @patch('json.dump')
# @patch('builtins.open', new_callable=MagicMock)
# def test_write_log(mock_open, mock_json_dump, controller):
#     log_path = 'path/to/log.json'
#     controller.write_log(log_path)
#     mock_open.assert_called_with(log_path, 'w')
#     # Verify that json.dump was called with the correct arguments
#     mock_json_dump.assert_called()
