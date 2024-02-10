# -*- coding: utf-8 -*-
import numpy as np
from PoliwhiRL.environment.RAM_locations import locations


def calc_rewards(
    controller,
    default_reward=0.01,
    use_sight=False,
):
    total_reward = -default_reward * 1

    if use_sight:
        if controller.is_new_vision():
            total_reward += default_reward * 2

    # Encourage getting out of location
    if controller.get_current_location() not in controller.locs:
        controller.locs.add(controller.get_current_location())
        if controller.get_current_location() in locations:
            total_reward += default_reward * 50
            controller.extend_timeout(250)
        else:
            total_reward += default_reward * 10
            controller.extend_timeout(10)

    # Encourage moving around
    cur_xy = controller.get_XY()
    if cur_xy not in controller.xy:
        total_reward += default_reward * 5
        controller.xy.add(cur_xy)

    # Encourage party pokemon
    total_level, total_hp, total_exp = controller.party_info()
    if total_level > np.sum(controller.max_total_level):
        total_reward += default_reward * 100
        controller.max_total_level = total_level
        controller.extend_timeout(500)

    # encourage max xp
    if total_exp > np.sum(controller.max_total_exp):
        total_reward += default_reward * 100
        controller.max_total_exp = total_exp
        controller.extend_timeout(100)

    if controller.pkdex_seen() > controller.max_pkmn_seen:
        total_reward += default_reward * 100
        controller.max_pkmn_seen = controller.pkdex_seen()
        controller.extend_timeout(100)

    if controller.pkdex_owned() > controller.max_pkmn_owned:
        if controller.pkdex_owned() == 1:
            #first time getting a pokemon lets allow a lot more exploration
            controller.extend_timeout(5000)
            print("Ding Ding Ding, we gotta pokemon!")
        total_reward += default_reward * 200
        controller.max_pkmn_owned = controller.pkdex_owned()
        controller.extend_timeout(200)

    if controller.get_player_money() > controller.max_money:
        total_reward += default_reward * 100
        controller.max_money = controller.get_player_money()
        controller.extend_timeout(100)
    elif controller.get_player_money() < controller.max_money:
        total_reward -= default_reward * 100
        controller.max_money = controller.get_player_money()

    if total_reward > 0:
        controller.extend_timeout(1)  # Encourage exploration

    return total_reward
