# -*- coding: utf-8 -*-
from PoliwhiRL.environment.controls import Controller as environment
from os.path import basename
from tqdm import tqdm


def explore(
    num_episodes, rom_path, state_path, episode_length, sight=False, parallel=False
):
    if parallel:
        raise NotImplementedError("Parallel exploration not yet implemented")
    savename = f"explore_{sight}_sight_state_{basename(state_path)}"
    print("use sight:", sight)
    env = environment(
        rom_path,
        state_path,
        episode_length,
        log_path=f"./logs/{savename}_log.json",
        use_sight=sight,
    )
    for i in tqdm(range(num_episodes), desc="Episodes..."):
        done = False
        while not done:
            img, reward, done = env.step(env.random_move())
            env.record(i, 0, savename)
            # print with carriage return
            print(
                f"Episode {i} | Reward {reward} | Steps {env.steps} | Current Timeout {env.timeout}",
                end="\r",
            )
        env.reset()
    env.close()
