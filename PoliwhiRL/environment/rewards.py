# -*- coding: utf-8 -*-
from skimage.metrics import structural_similarity as ssim
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PoliwhiRL.environment.RAM_locations import locations


def is_similar(target_image, image_list, threshold=99):
    def compare(target, other):
        similarity_index = ssim(target, other)
        return similarity_index * 100 >= threshold

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda img: compare(target_image, img), image_list)

    return any(results)


def calc_rewards(
    controller,
    default_reward=0.01,
    use_sight=False,
):
    total_reward = -default_reward * 1
    if use_sight:
        cur_img = np.array(controller.screen_image().convert("L"))
        if len(controller.imgs) > 0:
            if not is_similar(cur_img, controller.imgs):
                total_reward += default_reward * 2
                controller.imgs.append(cur_img)
        else:
            controller.imgs.append(cur_img)

    # Encourage getting out of location
    if controller.get_current_location() not in controller.locs:
        controller.locs.add(controller.get_current_location())
        if controller.get_current_location() in locations:
            total_reward += default_reward * 50
            controller.extend_timeout(50)

    # Encourage moving around
    cur_xy = controller.get_XY()
    if cur_xy not in controller.xy:
        total_reward += default_reward * 2
        controller.xy.add(cur_xy)
        controller.extend_timeout(1)

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

    return total_reward
