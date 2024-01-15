from PIL import Image
import torch 
import random
import numpy as np
import os 

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
    
def document(episode_id, step_id, img, button_press, reward, scale, grayscale):
    # for each episode we want to record a image of each step
    # as well as the button press that was made as part of the image name
    # each run should have its own directory 
    if not os.path.isdir("./runs"):
        os.mkdir("./runs")

    if not os.path.isdir("./runs/run_{}".format(episode_id)):
        os.mkdir("./runs/run_{}".format(episode_id))
    # save image 
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    # convert image scale and grayscale if needed
    if scale != 1:
        img = img.resize([int(s * scale) for s in img.size])
    if grayscale:
        img = img.convert("L")    
    img.save("./runs/run_{}/{}_{}_{}.png".format(episode_id, step_id, button_press, reward))


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

def save_checkpoint(checkpoint_path, model, optimizer, start_episode, epsilon):
    # Save checkpoint
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    checkpoint_path = checkpoint_path + "checkpoint_{}.pth".format(start_episode)
    print(f"Saving checkpoint to '{checkpoint_path}'")
    torch.save(
        {
            "start_episode": start_episode,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epsilon": epsilon,
        },
        checkpoint_path,
    )