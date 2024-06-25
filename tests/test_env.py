# -*- coding: utf-8 -*-
from PoliwhiRL.models.PPO.training_functions import create_environment


def test_create_environment():
    config = {
        "rom_path": "./emu_files/Pokemon - Crystal Version.gbc",
        "state_path": "./emu_files/states/start.state",
        "episode_length": 64,
        "N_goals_target": 1,
        # Add other necessary config parameters
    }

    env = create_environment(config)

    assert env is not None
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "close")
