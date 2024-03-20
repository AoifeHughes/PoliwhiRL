# -*- coding: utf-8 -*-
from PoliwhiRL.environment.controller import Controller
from PoliwhiRL.models.PPO.training_functions import train_ppo, load_latest_checkpoint
from .PPO import PPOModel


def setup_and_train_ppo(config):
    env = Controller(config)
    input_dim = (1 if config["use_grayscale"] else 3, *env.screen_size())
    output_dim = len(env.action_space)
    model = PPOModel(input_dim, output_dim).to(config["device"])
    start_episode = load_latest_checkpoint(model, config["checkpoint"])
    train_ppo(model, env, config, start_episode)
    env.close()
