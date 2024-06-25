# -*- coding: utf-8 -*-
import os
import torch
from .training_functions import (
    setup_environment_and_model,
    train,
    save_checkpoint,
    create_environment,
    get_input_dimensions,
    get_output_dimensions,
    create_model,
    setup_optimizer,
    setup_scheduler,
)


def setup_and_train_ppo(config):
    updated_vars = {}
    for var in ["episode_length", "num_episodes", "N_goals_target"]:
        # check if var is a string
        if isinstance(config[var], str) and "," in config[var]:
            updated_vars[var] = [int(i) for i in config[var].split(",")]
        else:
            updated_vars[var] = [int(config[var])]

    all_rewards = []
    all_losses = []

    for idx, v in enumerate(updated_vars["episode_length"]):
        config["episode_length"] = updated_vars["episode_length"][idx]
        config["num_episodes"] = updated_vars["num_episodes"][idx]
        config["N_goals_target"] = updated_vars["N_goals_target"][idx]

        rewards, losses, final_model_state = run_instance(config)
        all_rewards.extend(rewards)
        all_losses.extend(losses)

    return all_rewards, all_losses, final_model_state


def run_instance(config, shared_model=None):
    env = create_environment(config)
    input_dim = get_input_dimensions(env, config)
    output_dim = get_output_dimensions(env)
    config["num_actions"] = output_dim

    if shared_model is not None:
        model = create_model(input_dim, output_dim, config)
        if isinstance(shared_model, str) and os.path.exists(shared_model):
            model.load_state_dict(
                torch.load(shared_model, map_location=config["device"])
            )
        elif isinstance(shared_model, dict):
            model.load_state_dict(shared_model)
        elif isinstance(shared_model, torch.nn.Module):
            model.load_state_dict(shared_model.state_dict())
        else:
            print("Warning: Could not load shared model. Using a new model.")
        start_episode = 0
    else:
        model, start_episode = setup_environment_and_model(config)

    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer)

    rewards, losses = train(
        model, env, optimizer, scheduler, config, start_episode
    )  # Train the model

    save_checkpoint(
        model,
        config["checkpoint"],
        start_episode + config["num_episodes"],
    )
    env.close()

    return rewards, losses, model.state_dict()
