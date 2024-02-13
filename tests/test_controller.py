import pytest
from unittest.mock import patch, MagicMock
from PoliwhiRL.environment.controller import Controller  # Adjust the import according to your project structure
import json 

# Sample configuration to be returned by the mock
mock_config = {
    "rom_path": "path/to/mock/rom",
    "state_path": "path/to/mock/state"
}



@pytest.fixture
def controller():
    rom_path = mock_config["rom_path"]
    state_path = mock_config["state_path"]
    episode_length = 25
    sight = False
    return Controller(rom_path,
        state_path,
        timeout=episode_length,
        log_path="./logs/rainbow_env_eval.json",
        use_sight=sight)

def test_initialization_with_mocked_config(controller):
    # Test if the Controller uses values from the mocked config
    assert controller.rom_path == mock_config["rom_path"]
    assert controller.state_path == mock_config["state_path"]
    # Additional assertions can be added here to verify initialization with mocked config

def test_step_function_updates_state_correctly(controller):
    initial_steps = controller.steps
    controller.step(0)  # Assuming 0 corresponds to a valid action
    assert controller.steps == initial_steps + 1
    # Add more assertions to check other state changes as needed

@patch('os.path.isfile', return_value=True)
@patch('shutil.copy')
def test_set_state(mock_copy, mock_isfile, controller):
    state_file = 'path/to/new_state'
    controller.set_state(state_file)
    mock_copy.assert_called_with(state_file, controller.temp_dir)
    assert controller.state_path == state_file  # Or the result of mock_copy if different

@patch('json.dump')
@patch('builtins.open', new_callable=MagicMock)
def test_write_log(mock_open, mock_json_dump, controller):
    log_path = 'path/to/log.json'
    controller.write_log(log_path)
    mock_open.assert_called_with(log_path, 'w')
    # Verify that json.dump was called with the correct arguments
    mock_json_dump.assert_called()
