from PIL import Image
import torch 
import random
import numpy as np
import os 
import matplotlib.pyplot as plt

def image_to_tensor(image, device, SCALE_FACTOR=0.5, USE_GRAYSCALE=False):
    # Check if the image is already a PIL Image; if not, convert it
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Scale the image if needed
    if SCALE_FACTOR != 1:
        image = image.resize([int(s * SCALE_FACTOR) for s in image.size])

    # Convert to grayscale if needed
    if USE_GRAYSCALE:
        image = image.convert("L")

    # Convert the PIL image to a numpy array
    image = np.array(image)

    # Add an extra dimension for grayscale images
    if USE_GRAYSCALE:
        image = np.expand_dims(image, axis=2)

    # Convert to a PyTorch tensor, rearrange dimensions, normalize, and send to device
    image = (
        torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
        / 255
    )
    image = image.to(device)  # Sending tensor to the specified device

    return image

def select_action(state, epsilon, device, movements, model):
    if random.random() > epsilon:
        with torch.no_grad():
            return (
                model(state).max(1)[1].view(1, 1).to(device) )
    else:
        return torch.tensor(
            [[random.randrange(len(movements))]],
            dtype=torch.long,
            device=device,
        ) 
    
def document(episode_id, step_id, img, button_press, reward, scale, grayscale, timeout, epsilon, phase):
    # for each episode we want to record a image of each step
    # as well as the button press that was made as part of the image name
    # each run should have its own directory 
    fldr = "./runs_"+str(phase)
    if not os.path.isdir(fldr):
        os.mkdir(fldr)
    save_dir = f"./{fldr}/run_{timeout}_{episode_id}_{np.around(epsilon,2)}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # save image 
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img.save(f"{save_dir}/step_{step_id}_{button_press}_{np.around(reward,2)}.png")



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
    print(f"Saving results to '{results_path}'")
    with open(results_path, "w") as f:
        f.write(str(results))


def plot_best_attempts(results_path, episodes, phase, results):
    # Plot best attempts
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    results_path = results_path + f"best_attempts_{episodes}_{phase}.png"
    print(f"Saving plot to '{results_path}'")
    fig, ax = plt.subplots(1, figsize=(5,5), dpi=100)
    ax.plot(results)
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Best Attempt")
    fig.savefig(results_path, bbox_inches="tight")
    np.savetxt(results_path.replace(".png", ".csv"), results, delimiter=",")
    plt.close(fig)