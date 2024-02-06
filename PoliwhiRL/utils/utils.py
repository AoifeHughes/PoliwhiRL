# -*- coding: utf-8 -*-
from PIL import Image
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools


def image_to_tensor(image, device, SCALE_FACTOR=1, USE_GRAYSCALE=False):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    if SCALE_FACTOR != 1:
        image = image.resize([int(s * SCALE_FACTOR) for s in image.size])

    if USE_GRAYSCALE:
        image = image.convert("L")

    image = np.array(image)

    if USE_GRAYSCALE:
        # Expand dims for grayscale to ensure it has a channel dimension
        image = np.expand_dims(image, axis=0)  # Use axis=0 to represent the channel
    else:
        # Permute RGB to be in channel dimension
        image = np.transpose(image, (2, 0, 1))

    image = torch.from_numpy(image).to(torch.float32) / 255.0
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
        pass
    fldr = "./runs/" + str(phase) + "/"
    # check if all folders and subfolders exist
    if not os.path.isdir(fldr):
        os.mkdir(fldr)
    save_dir = f"./{fldr}/{timeout}_{episode_id}_{np.around(epsilon,2)}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # save image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img.save(
        f"{save_dir}/step_{step_id}_{button_press}_{np.around(reward,2)}_{location}.png"
    )


def load_checkpoint(checkpoint_path, model, optimizer, start_episode, epsilon):
    # Check for latest checkpoint in checkpoints folder

    tmp_path = checkpoint_path
    if os.path.isdir(checkpoint_path):
        checkpoints = os.listdir(checkpoint_path)
        checkpoints = [x for x in checkpoints if x.endswith(".pth")]
        if len(checkpoints) > 0:
            # sort checkpoints by last modified date
            checkpoints.sort(key=lambda x: os.path.getmtime(checkpoint_path + x))
            checkpoint_path = checkpoint_path + checkpoints[-1]
    else:
        os.mkdir(checkpoint_path)
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = checkpoint["start_episode"]
        epsilon = checkpoint["epsilon"]
        checkpoint_path = tmp_path
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
    return start_episode, epsilon


def save_checkpoint(checkpoint_path, model, optimizer, start_episode, epsilon, timeout):
    # Save checkpoint
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    checkpoint_path = checkpoint_path + f"checkpoint_{timeout}_{start_episode}.pth"
    torch.save(
        {
            "start_episode": start_episode,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epsilon": epsilon,
        },
        checkpoint_path,
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
