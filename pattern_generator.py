import os
import numpy as np
from PIL import Image


def generate_stripe_pattern(width, height, stripe_width):
    pattern = np.zeros((height, width), dtype=np.uint8)
    for x in range(0, width, stripe_width * 2):
        pattern[:, x : x + stripe_width] = 255
    return pattern


def save_image(image, filename, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    image.save(filepath)


# Function to generate and save multiple stripe patterns
def generate_and_save_vertical_patterns():
    base_directory = "StripePatterns"
    for i in range(1, 25):
        # Create subdirectory for each pattern set
        subdirectory = os.path.join(base_directory, f"Vertical_PatternSet")

        # Generate and save vertical stripe pattern
        vertical_stripe_pattern = generate_stripe_pattern(1, 1080, 10)
        vertical_stripe_image = Image.fromarray(vertical_stripe_pattern)
        vertical_stripe_image_resized = vertical_stripe_image.resize(
            (1920, 1080), Image.NEAREST
        )
        save_image(
            vertical_stripe_image_resized,
            f"vertical_stripe_pattern_{i}.bmp",
            subdirectory,
        )


def generate_and_save_horizontal_patterns():
    base_directory = "StripePatterns"
    for i in range(1, 25):
        # Create subdirectory for each pattern set
        subdirectory = os.path.join(base_directory, f"Horizontal_PatternSet")
        # Generate and save horizontal stripe pattern
        horizontal_stripe_pattern = generate_stripe_pattern(1920, 1, 10)
        horizontal_stripe_image = Image.fromarray(horizontal_stripe_pattern)
        horizontal_stripe_image_resized = horizontal_stripe_image.resize(
            (1920, 1080), Image.NEAREST
        )
        save_image(
            horizontal_stripe_image_resized,
            f"horizontal_stripe_pattern_{i}.bmp",
            subdirectory,
        )

    print("24 Stripe patterns saved successfully in the 'StripePatterns' directory.")


# Generate and save the stripe patterns
generate_and_save_vertical_patterns()
generate_and_save_horizontal_patterns()
