import numpy as np
from PIL import Image
import os

def generate_random_color_image(width, height):
    """Generate an image of random color."""
    color = np.random.randint(0, 256, (3,), dtype=np.uint8)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(image)

def save_images(num_images, width, height, directory):
    """Save a specified number of random color images."""
    for i in range(num_images):
        image = generate_random_color_image(width, height)
        image.save(f"{directory}/random_color_image_{i+1}.png")

# Parameters
num_images = 40
width = 1024
height = 1024
directory = "random_images"

os.makedirs(directory, exist_ok=True)

# Generate and save images
save_images(num_images, width, height, directory)

print(f'Saved {num_images} random color images to {os.path.abspath(directory)}')