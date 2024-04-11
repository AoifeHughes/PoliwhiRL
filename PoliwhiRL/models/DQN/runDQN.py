# -*- coding: utf-8 -*-
from .parallel_agent import ParallelDQNAgent
from .single_agent import DQNAgent
from PoliwhiRL.environment.controller import Controller as Env
from PoliwhiRL.environment.controller import action_space


def run_model(config, record_id=0, done_lim=1000):
    env = Env(config)
    config["state_size"] = (
        1 if config.get("use_grayscale", False) else 3,
        *env.screen_size(),
    )
    config["action_size"] = len(action_space)
    config["record_id"] = record_id

    if config["num_workers"] > 1:
        if config["device"] != "cpu":
            print(
                "Warning: ParallelDQNAgent only supports CPU training. Switching to CPU"
            )
            config["device"] = "cpu"
        agent = ParallelDQNAgent(config)
    else:
        agent = DQNAgent(config)
    try:
        agent.load("final_model.pth")
        print("Loaded model from final_model.pth")
    except FileNotFoundError as e:
        print(e)
        print("No model found, training from scratch")

    if config["num_workers"] > 1:
        agent.train(config["num_episodes"])
    else:
        agent.train(
            env, config["num_episodes"], config["random_episodes"], done_lim, record_id
        )

    env.close()
    agent.save("final_model.pth")
