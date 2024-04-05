# -*- coding: utf-8 -*-
from .training_functions import (
    setup_environment_and_model,
    train,
    save_checkpoint,
)


def setup_and_train_ppo(config):

    updated_vars = {}
    for var in ["episode_length", "sequence_length", "num_episodes"]:
        if "," in config[var]:
            updated_vars[var] = [int(i) for i in config[var].split(",")]
        else:
            updated_vars[var] = [int(config[var])]
    for idx, v in enumerate(updated_vars["episode_length"]):
        config["episode_length"] = updated_vars["episode_length"][idx]
        config["sequence_length"] = updated_vars["sequence_length"][idx]
        config["num_episodes"] = updated_vars["num_episodes"][idx]

        env, model, optimizer, start_episode = setup_environment_and_model(config)
        train(model, env, optimizer, config, start_episode)  # Train the model
        save_checkpoint(
            model,
            config["checkpoint"],
            start_episode + config["num_episodes"],
        )
        env.close()
