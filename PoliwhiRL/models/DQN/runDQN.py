# -*- coding: utf-8 -*-
from .parallel_agent import ParallelDQNAgent
from .single_agent import DQNAgent
from PoliwhiRL.environment.controller import Controller as Env
from PoliwhiRL.environment.controller import action_space


def populate_database(config):
    cur_device = config["device"]
    config["device"] = "cpu"
    print("Populating database with random episodes...")
    pop_agent = ParallelDQNAgent(config, workers_override=8)
    pop_agent.populate_db(config.get("random_episodes", 100))
    print("Database populated...")
    config["device"] = cur_device


def load_model(agent):
    try:
        agent.load("final_model.pth")
        print("Loaded model from final_model.pth")
    except FileNotFoundError:
        print("No model found, training from scratch")


def save_model(agent):
    agent.save("final_model.pth")


def run_parallel_training(agent, config):
    agent.train(config["num_episodes"])


def run_single_training(agent, env, config, record_id):
    agent.train(
        env,
        config["num_episodes"],
        config["random_episodes"],
        config["extend_threshold"],
        record_id,
    )


def run_model(config, record_id=0):
    env = Env(config)
    config["state_size"] = (
        1 if config.get("use_grayscale", False) else 3,
        *env.screen_size(),
    )
    config["action_size"] = len(action_space)
    config["record_id"] = record_id

    populate_database(config)
    if config["num_workers"] > 1:
        if config["device"] != "cpu":
            print(
                "Warning: ParallelDQNAgent only supports CPU training. Switching to CPU"
            )
            config["device"] = "cpu"
        agent = ParallelDQNAgent(config)
    else:
        agent = DQNAgent(config)

    load_model(agent)

    if config["num_workers"] > 1:
        run_parallel_training(agent, config)
    else:
        run_single_training(agent, env, config, record_id)

    env.close()
    save_model(agent)
