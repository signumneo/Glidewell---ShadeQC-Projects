import cv2
import numpy as np
import glob

# Updated calibration parameters based on 1080p resolution and 28 cm camera spacing
mtx_l = np.array([[1440, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)
dist_l = np.array(
    [[-0.09285014, -0.26333288, -0.0043878, 0.04053753, -2.31623397]], dtype=np.float64
)

mtx_r = np.array([[1440, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)
dist_r = np.array(
    [[-74.5919200, 4468.62804, 0.0367861014, -3.28082640, 1164.35377]], dtype=np.float64
)

# Translation vector between the cameras
T = np.array([[160], [0], [0]], dtype=np.float64)  # 28 cm translation along the x-axis

R = np.array(
    [
        [0.97451456, 0.15347141, -0.16360896],
        [-0.15293303, 0.98810771, 0.0159577],
        [0.16411232, 0.0094702, 0.9863962],
    ],
    dtype=np.float64,
)

# Compute the rectification transforms using stereoRectify
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, (1920, 1080), R, T
)

# Load images for the left and right cameras
images_left = glob.glob(
    r"C:\Users\jacob\OneDrive\Desktop\Jacob_Personal Projects\SLS\Captured Frames\Calibration\Left\left_*.jpg"
)
images_right = glob.glob(
    r"C:\Users\jacob\OneDrive\Desktop\Jacob_Personal Projects\SLS\Captured Frames\Calibration\Right\right_*.jpg"
)

# Ensure equal number of images are loaded
if len(images_left) != len(images_right):
    print("Mismatch in the number of left and right images.")
    print(f"Left images: {len(images_left)}, Right images: {len(images_right)}")
    exit()

print(f"Number of image pairs loaded: {len(images_left)}")

# Loop over each pair of images
for i, (img_left, img_right) in enumerate(zip(images_left, images_right)):
    print(f"Processing image pair {i + 1}:")
    print(f"Left image path: {img_left}")
    print(f"Right image path: {img_right}")

    # Read the images
    img_l = cv2.imread(img_left)
    img_r = cv2.imread(img_right)

    # Resize images to 1200x900
    img_l = cv2.resize(img_l, (800, 900))
    img_r = cv2.resize(img_r, (800, 900))

    # Rectify the images using the calculated rectification transforms
    map_x_l, map_y_l = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, img_l.shape[:2][::-1], cv2.CV_32FC1
    )
    map_x_r, map_y_r = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, img_r.shape[:2][::-1], cv2.CV_32FC1
    )

    rectified_img_l = cv2.remap(img_l, map_x_l, map_y_l, cv2.INTER_LINEAR)
    rectified_img_r = cv2.remap(img_r, map_x_r, map_y_r, cv2.INTER_LINEAR)

    # Combine the rectified images side by side
    combined_rectified = cv2.hconcat([rectified_img_l, rectified_img_r])

    # Resize the combined image to fit within 1700x900
    combined_rectified_resized = cv2.resize(combined_rectified, (1700, 900))

    # Draw epipolar lines
    num_lines = 50  # Number of epipolar lines to draw
    line_spacing = combined_rectified_resized.shape[0] // num_lines

    for i in range(1, num_lines):
        y = i * line_spacing
        cv2.line(
            combined_rectified_resized,
            (0, y),
            (combined_rectified_resized.shape[1], y),
            (0, 255, 0),
            1,
        )

    # Display the resized combined rectified image with epipolar lines
    cv2.imshow(
        "Combined Rectified Images with Epipolar Lines", combined_rectified_resized
    )
    cv2.waitKey(10000)

cv2.destroyAllWindows()
