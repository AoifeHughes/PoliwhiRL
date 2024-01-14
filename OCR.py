# -*- coding: utf-8 -*-
from PIL import Image
import pytesseract


def preprocess_image(image):
    # Scale the image by a factor of 2
    original_size = image.size
    scaled_size = tuple(2 * x for x in original_size)
    image = image.resize(scaled_size, Image.LANCZOS)
    # Convert to grayscale for better processing
    image = image.convert("L")
    # Apply threshold to get a binary image
    threshold_value = 120  # You might need to adjust this
    image = image.point(lambda p: p > threshold_value and 255)
    # Additional filters can be applied if necessary
    # image = image.filter(ImageFilter.MedianFilter())
    return image


def extract_text(image):
    text = pytesseract.image_to_string(image, config="--psm 11")
    return text
