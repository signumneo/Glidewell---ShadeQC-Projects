import numpy as np
import cv2
import os

# Resolution of the projector (DLP2010EVM)
width, height = 854, 480

# Number of patterns to generate
num_patterns = 5  # This can be adjusted based on your accuracy needs

# Create output directory
output_dir = "binary_patterns"
os.makedirs(output_dir, exist_ok=True)


def generate_vertical_pattern(pattern_index, width, height, num_patterns):
    """Generate vertical binary pattern"""
    pattern = np.zeros((height, width), dtype=np.uint8)
    stripe_width = width // (2 ** (pattern_index + 1))

    for i in range(2 ** (pattern_index + 1)):
        if i % 2 == 0:
            pattern[:, i * stripe_width : (i + 1) * stripe_width] = 255

    return pattern


def generate_horizontal_pattern(pattern_index, width, height, num_patterns):
    """Generate horizontal binary pattern"""
    pattern = np.zeros((height, width), dtype=np.uint8)
    stripe_height = height // (2 ** (pattern_index + 1))

    for i in range(2 ** (pattern_index + 1)):
        if i % 2 == 0:
            pattern[i * stripe_height : (i + 1) * stripe_height, :] = 255

    return pattern


# Generate and save vertical and horizontal patterns
for i in range(num_patterns):
    vertical_pattern = generate_vertical_pattern(i, width, height, num_patterns)
    horizontal_pattern = generate_horizontal_pattern(i, width, height, num_patterns)

    # Save vertical patterns
    vertical_filename = os.path.join(output_dir, f"vertical_pattern_{i + 1}.bmp")
    cv2.imwrite(vertical_filename, vertical_pattern)

    # Save horizontal patterns
    horizontal_filename = os.path.join(output_dir, f"horizontal_pattern_{i + 1}.bmp")
    cv2.imwrite(horizontal_filename, horizontal_pattern)

print(
    "Binary vertical and horizontal patterns have been generated and saved as BMP files."
)

