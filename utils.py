from PIL import Image
import torch 
import random
import numpy as np

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
    
