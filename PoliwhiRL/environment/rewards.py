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
            controller.extend_timeout(100)
        else:
            total_reward += default_reward * 10
            controller.extend_timeout(10)

    # Encourage moving around
    cur_xy = controller.get_XY()
    if cur_xy not in controller.xy:
        total_reward += default_reward * 2
        controller.xy.add(cur_xy)

    # Encourage party pokemon
    total_level, total_hp, total_exp = controller.party_info()
    if total_level > np.sum(controller.max_total_level):
        total_reward += default_reward * 100
        controller.max_total_level = total_level
        controller.extend_timeout(100)

    # encourage max xp
    if total_exp > np.sum(controller.max_total_exp):
        total_reward += default_reward * 100
        controller.max_total_exp = total_exp
        controller.extend_timeout(100)

    if total_reward > 0:
        controller.extend_timeout(1) # Encourage exploration

    return total_reward
