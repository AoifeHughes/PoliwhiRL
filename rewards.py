import numpy as np 
def calc_rewards(action, loc, visited_locations, controller, positive_keywords, negative_keywords, max_total_level, default_reward=0.01):
    # store if has been rewarded recently
    # if has been rewarded recently, then don't reward again

    total_reward = 0
    text = controller.get_text_on_screen()
    # check if any of the positive keywords are in the text
    for keyword in positive_keywords:
        if keyword in text.lower() and not positive_keywords[keyword]:
            positive_keywords[keyword] = True
            total_reward += default_reward

        else:
            positive_keywords[keyword] = False
    # check if any of the negative keywords are in the text
    for keyword in negative_keywords:
        if keyword in text.lower() and not negative_keywords[keyword]:
            negative_keywords[keyword] = True
            total_reward -= default_reward

        else:
            negative_keywords[keyword] = False

    # We should discourage start and select
    if action == "START" or action == "SELECT":
        total_reward -= default_reward * 2

    # Encourage exploration
    # if loc isn't in set then reward
    if loc not in visited_locations:
        total_reward += default_reward
        # add loc to visited locations
        visited_locations.add(loc)
    
    # Encourage party pokemon
    total_level, total_hp, total_exp= controller.party_info()

    if total_level > np.sum(max_total_level):
        total_reward += default_reward
        max_total_level[0] = total_level

    return total_reward