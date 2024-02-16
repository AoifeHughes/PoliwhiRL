# -*- coding: utf-8 -*-
import sys
from skimage.metrics import structural_similarity as ssim
import cv2


def compute_uniqueness_score(img1_path, img2_path):
    # Load the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Ensure images are the same size
    img1_gray = cv2.resize(img1_gray, (img2_gray.shape[1], img2_gray.shape[0]))

    # Compute SSIM between two images
    similarity = ssim(img1_gray, img2_gray)

    # Convert similarity to uniqueness
    uniqueness = 1 - similarity  # Directly invert similarity

    # Convert to percentage
    uniqueness_percentage = uniqueness * 100

    return uniqueness_percentage


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image1_path> <image2_path>")
        sys.exit(1)

    img1_path, img2_path = sys.argv[1], sys.argv[2]
    uniqueness_percentage = compute_uniqueness_score(img1_path, img2_path)

    print(f"Percentage Likelihood of Being Unique: {uniqueness_percentage:.2f}%")
