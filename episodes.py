# -*- coding: utf-8 -*-
from utils import image_to_tensor, select_action
from tqdm import tqdm
from itertools import count
from controls import Controller
from utils import document
from rewards import calc_rewards
import torch
import random


def explore_episode(rom_path, timeout, nsteps):
    controller = Controller(rom_path)
    movements = controller.movements
    locs = set()
    xy = set()
    imgs = []
    max_total_level = [0]
    max_total_exp = [0]
    states = []
    stored_states = 0
    for t in tqdm(range(timeout)):
        action = random.randrange(len(movements))
        controller.handleMovement(movements[action])
        img = controller.screen_image()
        reward = calc_rewards(
            controller,
            max_total_level,
            img,
            imgs,
            xy,
            locs,
            max_total_exp,
            default_reward=0.01,
        )

        states.append(controller.create_memory_state(controller))
        if reward > 0.1:
            savname = f"{stored_states}_reward_{reward}_loc_{controller.get_current_location()}_xy_{controller.get_XY()}.state"
            controller.store_state(states[0], savname)
            document(
                0,
                savname,
                img,
                movements[action],
                reward,
                timeout,
                1,
                "explore",
            )
            stored_states += 1
        if len(states) > nsteps:
            states.pop(0)


def run_episode(
    i,
    rom_path,
    state_path,
    model,
    epsilon,
    device,
    SCALE_FACTOR,
    USE_GRAYSCALE,
    timeout,
    n_steps=100,
    phase=0,
    document_mode=False,
):
    controller = Controller(rom_path, state_path)
    movements = controller.movements
    initial_img = controller.screen_image()
    state = image_to_tensor(initial_img, device, SCALE_FACTOR, USE_GRAYSCALE)
    done = False
    n_step_buffer = []
    n_step_buffers = []
    total_reward = 0
    max_total_level = [0]
    max_total_exp = 0
    locs = set()
    xy = set()
    imgs = []

    # Initial reward calculation with the first image
    _ = calc_rewards(
        controller,
        max_total_level,
        initial_img,
        imgs,
        xy,
        locs,
        max_total_exp,
        default_reward=0.01,
    )

    for t in count():
        action = select_action(state, epsilon, device, movements, model)
        controller.handleMovement(movements[action.item()])
        img = controller.screen_image()
        reward = calc_rewards(
            controller,
            max_total_level,
            img,
            imgs,
            xy,
            locs,
            max_total_exp,
            default_reward=0.01,
        )

        action_tensor = torch.tensor([action], dtype=torch.int64, device=device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
        next_state = (
            image_to_tensor(img, device, SCALE_FACTOR, USE_GRAYSCALE)
            if not done
            else None
        )

        n_step_buffer.append((state, action_tensor, reward_tensor, next_state))

        if len(n_step_buffer) == n_steps or done:
            n_step_buffers.append(
                list(n_step_buffer)
            )  # Shallow copy to avoid deep copy overhead
            n_step_buffer.clear()  # Clear the buffer for new data

        state = next_state
        total_reward += reward
        if done or (timeout > 0 and t >= timeout):
            break
        if document_mode:
            document(
                i,
                t,
                img,
                movements[action.item()],
                reward,
                timeout,
                epsilon,
                phase,
            )

    controller.stop(save=False)

    return n_step_buffers, total_reward
