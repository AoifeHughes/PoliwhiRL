# -*- coding: utf-8 -*-
import numpy as np
from PoliwhiRL.environment.RAM_locations import locations


def update_for_locations(controller, total_reward, default_reward):
    if controller.get_current_location() not in controller.locations:
        controller.locations.add(controller.get_current_location())
        if controller.get_current_location() == locations["MrPokemonsHouse"]:
            total_reward += default_reward * 100
            controller.reset_has_reached_reward_locations_xy()
            try:
                controller.store_controller_state("./PerfectRunState.pkl")
                print("Made it to Mr. Pokemon's House")
            except Exception as e:
                print(e)
                print("Could not save state")

    return total_reward


def update_for_vision(controller, total_reward, default_reward):
    if controller.is_new_vision():
        total_reward += default_reward * 5
    return total_reward


def update_for_party_pokemon(controller, total_reward, default_reward):
    total_level, total_hp, total_exp = controller.party_info()
    if total_level > np.sum(controller.max_total_level):
        total_reward += default_reward * 200
        controller.max_total_level = total_level

    if total_hp > np.sum(controller.max_total_hp):
        total_reward += default_reward * 100
        controller.max_total_hp = total_hp

    if total_exp > np.sum(controller.max_total_exp):
        total_reward += default_reward * 100
        controller.max_total_exp = total_exp

    return total_reward


def update_for_movement(controller, total_reward, default_reward):
    cur_xy = controller.get_XY()
    if cur_xy not in controller.xy:
        total_reward += default_reward * 10
        controller.xy.add(cur_xy)
    return total_reward


def update_for_pokedex(controller, total_reward, default_reward):
    if controller.pkdex_seen() > controller.max_pkmn_seen:
        total_reward += default_reward * 100
        controller.max_pkmn_seen = controller.pkdex_seen()

    if controller.pkdex_owned() > controller.max_pkmn_owned:
        total_reward += default_reward * 200
        controller.max_pkmn_owned = controller.pkdex_owned()
    return total_reward


def update_for_money(controller, total_reward, default_reward):
    if controller.get_player_money() > controller.max_money:
        total_reward += default_reward * 100
        controller.max_money = controller.get_player_money()
    elif controller.get_player_money() < controller.max_money:
        total_reward -= default_reward * 100
        controller.max_money = controller.get_player_money()
    return total_reward


def update_for_xy_checkpoints(controller, total_reward, default_reward):
    if controller.get_current_location() in controller.has_reached_reward_locations_xy:
        # we are in a location we want to be in now check xy
        if (
            controller.get_XY()
            in controller.has_reached_reward_locations_xy[
                controller.get_current_location()
            ]
        ):
            if (
                controller.has_reached_reward_locations_xy[
                    controller.get_current_location()
                ][controller.get_XY()]
                is True
            ):
                return total_reward
            else:
                print("Reached a new named XY checkpoint")
                print("Checkpoint reached at: ", controller.get_XY())
                controller.has_reached_reward_locations_xy[
                    controller.get_current_location()
                ][controller.get_XY()] = True
                controller.extend_timeout(500)
                return total_reward + default_reward * 100
    return total_reward


def calc_rewards(controller, default_reward=0.01, use_sight=False):
    total_reward = -default_reward * 1

    if use_sight:
        total_reward = update_for_vision(controller, total_reward, default_reward)

    for func in [
        update_for_party_pokemon,
        update_for_movement,
        update_for_pokedex,
        update_for_money,
        update_for_xy_checkpoints,
        update_for_locations,
    ]:
        total_reward = func(controller, total_reward, default_reward)

    return total_reward
