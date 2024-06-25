# -*- coding: utf-8 -*-
import multiprocessing as mp
import torch
import os
import numpy as np

from PoliwhiRL.utils.utils import plot_best_attempts
from .PPO_run import run_instance
from .training_functions import (
    create_environment,
    get_input_dimensions,
    get_output_dimensions,
    create_model,
    load_latest_checkpoint,
)


def run_ppo_instance(config, process_id, shared_model_path, iteration):
    # Set device to MPS for this process
    config["device"] = torch.device("mps")

    # Modify config to save results in process-specific directories
    config["checkpoint"] = f"checkpoints/process_{process_id}"
    config["result_dir"] = f"results/process_{process_id}"
    config["agent"] = f"process_{process_id}"
    config["iteration"] = iteration

    # Create environment and get dimensions
    env = create_environment(config)
    input_dim = get_input_dimensions(env, config)
    output_dim = get_output_dimensions(env)
    config["num_actions"] = output_dim

    # Create model
    model = create_model(input_dim, output_dim, config)

    # Load shared weights
    if isinstance(shared_model_path, str) and os.path.exists(shared_model_path):
        model.load_state_dict(
            torch.load(shared_model_path, map_location=config["device"])
        )
    elif isinstance(shared_model_path, dict):
        model.load_state_dict(shared_model_path)
    else:
        print(f"Warning: Could not load shared model from {shared_model_path}")

    # Run PPO training
    rewards, losses, final_model_state = run_instance(config, model)
    os.makedirs(f"results/process_{process_id}", exist_ok=True)

    # Save results
    torch.save(
        {"rewards": rewards, "losses": losses, "model_state": final_model_state},
        f"results/process_{process_id}/final_results.pth",
    )


def run_multi(config):
    num_goals_max = config.get("num_goals_max", 3)
    num_processes = config.get("num_processes", 4)
    num_iterations = config.get("num_iterations", 10)

    # Initialize the shared model
    env = create_environment(config)
    input_dim = get_input_dimensions(env, config)
    output_dim = get_output_dimensions(env)
    config["num_actions"] = output_dim
    initial_model = create_model(input_dim, output_dim, config)
    shared_model_path = "models/shared_model.pth"
    torch.save(initial_model.state_dict(), shared_model_path)
    for goal_max in range(num_goals_max):
        config["N_goals_target"] = goal_max + 1
        for iteration in range(num_iterations):
            print(f"Starting iteration {iteration + 1}/{num_iterations}")

            processes = []
            for i in range(num_processes):
                p = mp.Process(
                    target=run_ppo_instance,
                    args=(config, i, shared_model_path, iteration),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # After all processes are done, aggregate results and update the shared model
            avg_rewards, avg_losses = aggregate_and_update_model(
                num_processes, config, shared_model_path
            )
            print(
                f"Target Goals: {goal_max},  Iteration {iteration + 1} completed. Average Rewards: {avg_rewards}, Average Losses: {avg_losses}"
            )


def aggregate_and_update_model(num_processes, config, shared_model_path):
    all_rewards = []
    all_losses = []
    aggregated_state_dict = None

    for i in range(num_processes):
        results = torch.load(f"results/process_{i}/final_results.pth")
        all_rewards.extend(results["rewards"])
        all_losses.extend(results["losses"])

        if aggregated_state_dict is None:
            aggregated_state_dict = results["model_state"]
        else:
            # Average the model parameters
            for key in aggregated_state_dict:
                aggregated_state_dict[key] += results["model_state"][key]

    # Compute the average of the model parameters
    for key in aggregated_state_dict:
        aggregated_state_dict[key] /= num_processes

    # Create a new model with the averaged parameters
    env = create_environment(config)
    input_dim = get_input_dimensions(env, config)
    output_dim = get_output_dimensions(env)
    updated_model = create_model(input_dim, output_dim, config)
    updated_model.load_state_dict(aggregated_state_dict)

    # Save the updated model
    torch.save(updated_model.state_dict(), shared_model_path)

    plot_best_attempts("results/multi", "multi", "all", all_rewards)

    avg_rewards = np.mean(all_rewards)
    avg_losses = np.mean(all_losses)

    print(f"Average Rewards: {avg_rewards}")
    print(f"Average Losses: {avg_losses}")

    return avg_rewards, avg_losses


def setup_environment_and_model(config):
    env = create_environment(config)
    input_dim = get_input_dimensions(env, config)
    output_dim = get_output_dimensions(env)
    config["num_actions"] = output_dim

    model = create_model(input_dim, output_dim, config)
    start_episode = load_latest_checkpoint(model, config["checkpoint"])

    return model, start_episode
