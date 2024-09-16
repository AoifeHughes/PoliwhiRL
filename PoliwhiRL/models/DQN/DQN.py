# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.models.DQN.agents.prime_agent import PokemonAgent


def setup_and_train(config):
    env = Env(config)
    state_shape = (
        env.get_screen_size() if config["vision"] else env.get_game_area().shape
    )
    num_actions = env.action_space.n

    agent = PokemonAgent(state_shape, num_actions, config)
    agent.run_ciriculum(1, config['N_goals_target'], 200)
