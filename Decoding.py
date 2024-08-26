import cv2
import numpy as np
import glob
import open3d as o3d
import matplotlib.pyplot as plt

# Updated calibration parameters based on 1080p resolution and 28 cm camera spacing
mtx_l = np.array([[1440, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)
dist_l = np.array(
    [-0.01013339, 0.08788255, 0.01637522, 0.03001656, -0.08250072], dtype=np.float64
)

mtx_r = np.array([[1440, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)
dist_r = np.array(
    [-0.13027452, 0.17510251, 0.00224723, 0.02801563, -0.00785824], dtype=np.float64
)

# Translation vector between the cameras
T = np.array([[280], [0], [0]], dtype=np.float64)  # 28 cm translation along the x-axis

# 5-degree tilt around the y-axis
theta = 5 * np.pi / 180  # 5 degrees in radians
R = np.array(
    [
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ],
    dtype=np.float64,
)

# Stereo rectification
image_size = (640, 480)  # assuming 1080p resolution
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, image_size, R, T
)

# Compute rectification maps
mapx_l, mapy_l = cv2.initUndistortRectifyMap(
    mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
)
mapx_r, mapy_r = cv2.initUndistortRectifyMap(
    mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
)


# Load images into dictionaries
def load_images(image_paths):
    images = {}
    for path in image_paths:
        image_name = path.split("\\")[-1]  # Extract image name from the path
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {path}")
        else:
            images[image_name] = image
    return images


# Vertical pattern images
images_left_vertical = load_images(
    sorted(
        glob.glob(r"C:\Users\jacob\Downloads\SLS\New Frames\Vertical\Left\left_*.jpg")
    )
)
images_right_vertical = load_images(
    sorted(
        glob.glob(r"C:\Users\jacob\Downloads\SLS\New Frames\Vertical\Right\right_*.jpg")
    )
)

# Horizontal pattern images
images_left_horizontal = load_images(
    sorted(
        glob.glob(
            r"CC:\Users\jacob\Downloads\SLS\New Frames\Horizontal\Left\left_*.jpg"
        )
    )
)
images_right_horizontal = load_images(
    sorted(
        glob.glob(
            r"C:\Users\jacob\Downloads\SLS\New Frames\Horizontal\Right\right_*.jpg"
        )
    )
)


# Function to display images in the same window
def display_combined_images(images_left, images_right, window_name):
    for img_name in images_left:
        img_l = images_left[img_name]
        img_r = images_right.get(img_name.replace("left_", "right_"))

        if img_r is None:
            print(f"Failed to find corresponding right image for {img_name}")
            continue

        combined = cv2.hconcat(
            [img_l, img_r]
        )  # Combine left and right images side by side
        cv2.imshow(window_name, combined)
        cv2.waitKey(2000)  # Display for 2 seconds

    cv2.destroyAllWindows()


# Function to print a message to the console
def print_message(message):
    print(f"\n[INFO] {message}")


# Phase 1: Load and Display Original Images
print_message("Starting Phase 1: Loading and displaying original images")

# Display vertical pattern images
display_combined_images(
    images_left_vertical, images_right_vertical, "Combined Vertical Patterns"
)

# Display horizontal pattern images
display_combined_images(
    images_left_horizontal, images_right_horizontal, "Combined Horizontal Patterns"
)

print_message("Phase 1 completed: Images loaded and displayed successfully")

# Wait for user confirmation to proceed
input(
    "\nPress Enter to start Phase 2: Image rectification and disparity calculation..."
)

# Phase 2: Rectification and Disparity Computation
print_message("Starting Phase 2: Rectification and disparity computation")

# Initialize an array to store decoded disparities
disparities_vertical = []
disparities_horizontal = []

# Process vertical patterns
for img_name in images_left_vertical:
    img_l = images_left_vertical[img_name]
    img_r = images_right_vertical.get(img_name.replace("left_", "right_"))

    if img_r is None:
        print(f"Failed to find corresponding right image for {img_name}")
        continue

    rectified_l = cv2.remap(img_l, mapx_l, mapy_l, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(img_r, mapx_r, mapy_r, cv2.INTER_LINEAR)

    combined_rectified = cv2.hconcat([rectified_l, rectified_r])
    cv2.imshow("Combined Rectified Images", combined_rectified)
    cv2.waitKey(2000)  # Display for 2 seconds

    disparity = cv2.absdiff(rectified_l, rectified_r)
    _, disparity = cv2.threshold(disparity, 50, 255, cv2.THRESH_BINARY)
    disparities_vertical.append(disparity)

# Process horizontal patterns
for img_name in images_left_horizontal:
    img_l = images_left_horizontal[img_name]
    img_r = images_right_horizontal.get(img_name.replace("left_", "right_"))

    if img_r is None:
        print(f"Failed to find corresponding right image for {img_name}")
        continue

    rectified_l = cv2.remap(img_l, mapx_l, mapy_l, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(img_r, mapx_r, mapy_r, cv2.INTER_LINEAR)

    combined_rectified = cv2.hconcat([rectified_l, rectified_r])
    cv2.imshow("Combined Rectified Images", combined_rectified)
    cv2.waitKey(2000)  # Display for 2 seconds

    disparity = cv2.absdiff(rectified_l, rectified_r)
    _, disparity = cv2.threshold(disparity, 50, 255, cv2.THRESH_BINARY)
    disparities_horizontal.append(disparity)

cv2.destroyAllWindows()

# Ensure that disparity arrays are not empty before proceeding
if len(disparities_vertical) == 0 or len(disparities_horizontal) == 0:
    print("Error: Disparities could not be computed.")
    exit()

# Combine vertical and horizontal disparity maps by averaging and convert to a proper type
disp_combined = (
    disparities_vertical[0].astype(np.float32)
    + disparities_horizontal[0].astype(np.float32)
) / 2

# Ensure the disparity map is of type CV_32F before reprojecting
disp_combined = disp_combined.astype(np.float32)

# Display the combined disparity map
print_message("Displaying the combined disparity map")
plt.figure(figsize=(10, 7))
plt.imshow(disp_combined, cmap="gray")
plt.title("Combined Disparity Map")
plt.show()

# Wait for user confirmation to proceed
input("\nPress Enter to start Phase 3: Point cloud generation...")

# Phase 3: Generate the Point Cloud
print_message("Starting Phase 3: Point cloud generation")

# Generate the point cloud using triangulation
points_3D = cv2.reprojectImageTo3D(disp_combined, Q)

# Mask to remove invalid points
mask = disp_combined > disp_combined.min()
output_points = points_3D[mask]

# Convert points to Open3D point cloud format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(output_points)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# Optionally, save the point cloud to a file
o3d.io.write_point_cloud("output_point_cloud_combined.ply", pcd)

print_message("Point cloud generation completed")
