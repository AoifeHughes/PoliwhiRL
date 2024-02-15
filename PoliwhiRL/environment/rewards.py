# -*- coding: utf-8 -*-
import numpy as np
from PoliwhiRL.environment.RAM_locations import locations


def update_for_locations(controller, total_reward, default_reward):
    if controller.get_current_location() not in controller.locations:
        controller.locations.add(controller.get_current_location())
        if controller.get_current_location() == locations["MrPokemonsHouse"]:
            total_reward += default_reward * 100
            controller.extend_timeout(2000)
            controller.reset_has_reached_reward_locations_xy()
            controller.set_save_on_reset()
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
        controller.extend_timeout(500)

    if total_hp > np.sum(controller.max_total_hp):
        total_reward += default_reward * 100
        controller.max_total_hp = total_hp
        controller.extend_timeout(100)

    if total_exp > np.sum(controller.max_total_exp):
        total_reward += default_reward * 100
        controller.max_total_exp = total_exp
        controller.extend_timeout(100)

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
        controller.extend_timeout(250)

    if controller.pkdex_owned() > controller.max_pkmn_owned:
        if controller.pkdex_owned() == 1:
            controller.extend_timeout(1000)
            controller.set_save_on_reset()
        total_reward += default_reward * 200
        controller.max_pkmn_owned = controller.pkdex_owned()
        controller.extend_timeout(200)
    return total_reward


def update_for_money(controller, total_reward, default_reward):
    if controller.get_player_money() > controller.max_money:
        total_reward += default_reward * 100
        controller.max_money = controller.get_player_money()
        controller.extend_timeout(100)
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
            # check if been rewarded for this location_xy
            if (
                controller.has_reached_reward_locations_xy[
                    controller.get_current_location()
                ][controller.get_XY()]
                is True
            ):
                return total_reward
            else:
                controller.has_reached_reward_locations_xy[
                    controller.get_current_location()
                ][controller.get_XY()] = True
                controller.extend_timeout(300)
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

    if total_reward > 0:
        controller.extend_timeout(2)
    return total_reward
