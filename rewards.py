from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def is_similar(target_image, image_list, threshold=99):
    def compare(target, other):
        similarity_index = ssim(target, other)
        return similarity_index * 100 >= threshold

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda img: compare(target_image, img), image_list)

    return any(results)

def calc_rewards(controller, max_total_level, cur_img, imgs, default_reward=0.01):
    # store if has been rewarded recently
    # if has been rewarded recently, then don't reward again

    total_reward = -default_reward
    cur_img = np.array(cur_img.convert('L'))
    if len(imgs) > 0:
        if not is_similar(cur_img, imgs):
            total_reward += default_reward * 5
            imgs.append(cur_img)
    else:
        imgs.append(cur_img)

    # Encourage party pokemon
    total_level, total_hp, total_exp= controller.party_info()
    if total_level > np.sum(max_total_level):
        total_reward += default_reward * 10
        max_total_level[0] = total_level

    return total_reward