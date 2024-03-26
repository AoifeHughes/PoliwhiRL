# -*- coding: utf-8 -*-
from .training_functions import (
    setup_environment_and_model,
    train,
    save_checkpoint,
)


def setup_and_train_ppo(config):
    env, model, optimizer, start_episode = setup_environment_and_model(config)
    train(model, env, optimizer, config, start_episode)  # Train the model
    save_checkpoint(
        model,
        config["checkpoint"],
        start_episode + config["num_episodes"],
    )
    env.close()
