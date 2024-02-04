# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compare_images_pil(image1, image2):
    # Convert PIL images to numpy array
    image1_np = np.array(image1)
    image2_np = np.array(image2)

    # Convert images to grayscale if they are not
    if image1_np.ndim == 3:
        image1_np = image1_np.mean(axis=2)
    if image2_np.ndim == 3:
        image2_np = image2_np.mean(axis=2)

    # Resize images to match sizes if they're different
    if image1_np.shape != image2_np.shape:
        image2_np = np.array(image2.resize(image1.size))

    # Compute SSIM between the two images
    similarity_index = ssim(image1_np, image2_np)

    # Convert to percentage
    percentage_similarity = similarity_index * 100

    return percentage_similarity


# Example usage
image1 = Image.open("img1.png")
image2 = Image.open("img2.png")
similarity_percentage = compare_images_pil(image1, image2)
print(f"Similarity: {similarity_percentage:.2f}%")
