# -*- coding: utf-8 -*-
from PIL import Image
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools


def image_to_tensor(image, device):
    image = torch.from_numpy(np.transpose(image, (2, 0, 1))).to(torch.uint8) / 255.0
    image = image.to(device)

    return image


def select_action(state, epsilon, device, movements, model):
    if random.random() > epsilon:
        with torch.no_grad():
            return model(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(len(movements))]],
            dtype=torch.long,
            device=device,
        )


def document(
    episode_id, step_id, img, button_press, reward, timeout, epsilon, phase, location
):
    try:
        if not os.path.isdir("./runs"):
            os.mkdir("./runs")
    except Exception as e:
        print(e)
    fldr = "./runs/" + str(phase) + "/"
    # check if all folders and subfolders exist
    if not os.path.isdir(fldr):
        os.mkdir(fldr)
    save_dir = f"./{fldr}/{episode_id}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # save image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img.save(
        f"{save_dir}/step_{step_id}_btn_{button_press}_reward_{np.around(reward,2)}_ep_{np.around(epsilon,2)}_loc_{location}_timeout_{timeout}.png"
    )


def save_results(results_path, episodes, results):
    # Save results
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    results_path = results_path + f"results_{episodes}.txt"
    with open(results_path, "w") as f:
        f.write(str(results))


def plot_best_attempts(results_path, episodes, phase, results):
    # Ensure the results directory exists
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, f"best_attempts_{episodes}_{phase}.png")

    # Calculate cumulative mean
    cum_mean = np.cumsum(results) / np.arange(1, len(results) + 1)

    # Create plot
    fig, ax = plt.subplots(1, figsize=(10, 6), dpi=100)
    ax.plot(cum_mean, label="Cumulative Mean", color="blue", linewidth=2)

    ax.set_title("Performance Over Episodes")
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Average Reward")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    fig.tight_layout()

    fig.savefig(results_path)
    np.savetxt(results_path.replace(".png", ".csv"), results, delimiter=",")

    plt.close(fig)


def log_rewards(batch_rewards):
    return f"Average reward for last batch: {np.mean(batch_rewards)} | Best reward: {np.max(batch_rewards)}"


def chunked_iterable(iterable, size):
    it = iter(iterable)
    for _ in range(0, len(iterable), size):
        yield tuple(itertools.islice(it, size))
