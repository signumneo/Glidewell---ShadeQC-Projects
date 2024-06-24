import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_box_outline_points(length, width, height, density=10):
    """
    Generate a point cloud for the outline of a box with given dimensions and density.

    Parameters:
    - length: Length of the box
    - width: Width of the box
    - height: Height of the box
    - density: Number of points along each edge

    Returns:
    - points: A numpy array of shape (N, 3) representing the point cloud
    """
    x = np.linspace(0, length, density)
    y = np.linspace(0, width, density)
    z = np.linspace(0, height, density)

    # Create points on the six faces of the box
    faces = []

    # Front and back faces
    xx, yy = np.meshgrid(x, y)
    faces.append(np.vstack((xx.ravel(), yy.ravel(), np.zeros_like(xx).ravel())).T)
    faces.append(
        np.vstack((xx.ravel(), yy.ravel(), np.full_like(xx, height).ravel())).T
    )

    # Left and right faces
    yy, zz = np.meshgrid(y, z)
    faces.append(np.vstack((np.zeros_like(yy).ravel(), yy.ravel(), zz.ravel())).T)
    faces.append(
        np.vstack((np.full_like(yy, length).ravel(), yy.ravel(), zz.ravel())).T
    )

    # Top and bottom faces
    xx, zz = np.meshgrid(x, z)
    faces.append(np.vstack((xx.ravel(), np.zeros_like(xx).ravel(), zz.ravel())).T)
    faces.append(np.vstack((xx.ravel(), np.full_like(xx, width).ravel(), zz.ravel())).T)

    # Combine all face points
    points = np.vstack(faces)

    return points


def plot_point_cloud(points):
    """
    Plot a 3D point cloud.

    Parameters:
    - points: A numpy array of shape (N, 3) representing the point cloud
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    plt.show()


# Example usage:
length = 8
width = 3
height = 2
density = 20

points = generate_box_outline_points(length, width, height, density)
plot_point_cloud(points)
