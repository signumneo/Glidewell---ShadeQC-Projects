import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_ply(filename):
    """Read a PLY file and return points and colors as numpy arrays."""
    with open(filename, "r") as file:
        lines = file.readlines()

    points = []
    colors = []
    header = True
    for line in lines:
        if header:
            if line.strip() == "end_header":
                header = False
            continue

        parts = line.strip().split()
        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        colors.append([int(parts[3]), int(parts[4]), int(parts[5])])

    return np.array(points), np.array(colors)


# Path to the point cloud files
point_cloud_folder = r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python"
point_cloud_files = [f"point_cloud_{i:02d}.ply" for i in range(1, 31)]

# Initialize lists to store combined points and colors
combined_points = []
combined_colors = []

# Read and combine all point clouds
for filename in point_cloud_files:
    points, colors = read_ply(f"{point_cloud_folder}/{filename}")
    combined_points.append(points)
    combined_colors.append(colors)

# Convert lists to numpy arrays
combined_points = np.vstack(combined_points)
combined_colors = np.vstack(combined_colors)

# Reduce the number of points for visualization to prevent "not responding" issues
sample_indices = np.random.choice(combined_points.shape[0], size=100000, replace=False)
sampled_points = combined_points[sample_indices]
sampled_colors = combined_colors[sample_indices]

# Visualize the combined point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    sampled_points[:, 0],
    sampled_points[:, 1],
    sampled_points[:, 2],
    c=sampled_colors / 255.0,
    s=0.1,
)
plt.show()
