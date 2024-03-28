# -*- coding: utf-8 -*-
from .training_functions import (
    setup_environment_and_model,
    train,
    save_checkpoint,
)


def setup_and_train_ppo(config):
    if "," in config["episode_length"]:
        ep_lengths = [int(i) for i in config["episode_length"].split(",")]
    else:
        ep_lengths = [int(config["episode_length"])]
    for ep_length in ep_lengths:
        config["episode_length"] = ep_length
        env, model, optimizer, start_episode = setup_environment_and_model(config)
        train(model, env, optimizer, config, start_episode)  # Train the model
        save_checkpoint(
            model,
            config["checkpoint"],
            start_episode + config["num_episodes"],
        )
        env.close()
