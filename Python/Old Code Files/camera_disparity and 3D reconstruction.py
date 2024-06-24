import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load calibration data
mtx_left = np.load(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\camera_mtx_left.npy"
)
dist_left = np.load(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\camera_dist_left.npy"
)
mtx_right = np.load(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\camera_mtx_right.npy"
)
dist_right = np.load(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\camera_dist_right.npy"
)
R = np.load(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\R.npy"
)
T = np.load(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\T.npy"
)
Q = np.load(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\Q.npy"
)


# Compute rectification transforms
image_height, image_width = cv2.imread(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_left_1.jpg",
    cv2.IMREAD_GRAYSCALE,
).shape
R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right, (image_width, image_height), R, T
)

# Initialize lists to store point clouds
all_points_3D = []

for i in range(1, 31):
    img_left = cv2.imread(
        rf"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_left_{i}.jpg",
        cv2.IMREAD_GRAYSCALE,
    )
    img_right = cv2.imread(
        rf"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_right_{i}.jpg",
        cv2.IMREAD_GRAYSCALE,
    )

    # Apply rectification
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        mtx_left, dist_left, R1, P1, (image_width, image_height), cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        mtx_right, dist_right, R2, P2, (image_width, image_height), cv2.CV_16SC2
    )

    rectified_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

    # Save rectified images (optional, for verification)
    cv2.imwrite(f"rectified_left{i:02d}.jpg", rectified_left)
    cv2.imwrite(f"rectified_right{i:02d}.jpg", rectified_right)

    # Compute disparity using StereoSGBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 5,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disparity = stereo.compute(rectified_left, rectified_right)

    # Normalize disparity for visualization (optional)
    disparity_normalized = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    cv2.imwrite(f"disparity_map{i:02d}.jpg", disparity_normalized)

    # Reproject to 3D
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(rectified_left, cv2.COLOR_GRAY2BGR)
    mask = disparity > disparity.min()

    out_points = points_3D[mask]
    out_colors = colors[mask]

    all_points_3D.append((out_points, out_colors))


# Save all point clouds to PLY files (optional)
def save_point_cloud(filename, points, colors):
    with open(filename, "w") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write(
            "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        file.write("end_header\n")
        for p, c in zip(points, colors):
            file.write(f"{p[0]} {p[1]} {p[2]} {c[2]} {c[1]} {c[0]}\n")


for i, (points, colors) in enumerate(all_points_3D):
    save_point_cloud(f"point_cloud_{i+1:02d}.ply", points, colors)

# Combine point clouds for visualization
combined_points = np.vstack([points for points, _ in all_points_3D])
combined_colors = np.vstack([colors for _, colors in all_points_3D])

# Reduce the number of points for visualization to prevent "not responding" issues
sample_indices = np.random.choice(combined_points.shape[0], size=100000, replace=False)
sampled_points = combined_points[sample_indices]
sampled_colors = combined_colors[sample_indices]

# Visualize the combined point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    combined_points[:, 0],
    combined_points[:, 1],
    combined_points[:, 2],
    c=combined_colors / 255.0,
    s=0.1,
)
plt.show()
