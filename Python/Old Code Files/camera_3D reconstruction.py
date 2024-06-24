import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to triangulate points using projection matrices
def triangulate(mtx1, mtx2, R, T, points_left, points_right):
    # Convert points to numpy arrays
    uvs1 = np.array(points_left, dtype=float)
    uvs2 = np.array(points_right, dtype=float)

    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2

    def DLT(P1, P2, point1, point2):
        A = [
            point1[1] * P1[2, :] - P1[1, :],
            P1[0, :] - point1[0] * P1[2, :],
            point2[1] * P2[2, :] - P2[1, :],
            P2[0, :] - point2[0] * P2[2, :],
        ]
        A = np.array(A).reshape((4, 4))
        B = A.transpose() @ A
        U, s, Vh = np.linalg.svd(B, full_matrices=False)
        return Vh[3, 0:3] / Vh[3, 3]

    p3ds = [DLT(P1, P2, uv1, uv2) for uv1, uv2 in zip(uvs1, uvs2)]
    return np.array(p3ds)


# Load calibration parameters
left_camera_mtx = np.load("camera_mtx_left.npy")
right_camera_mtx = np.load("camera_mtx_right.npy")
R = np.load("R.npy")
T = np.load("T.npy")

# Corresponding points from the left and right images
points_left = [
    (206, 958),
    (1038, 949),
    (203, 1465),
    (1041, 1458),
    (1083, 992),
    (1086, 1440),
]
points_right = [(69, 905), (921, 912), (72, 1434), (931, 1428), (34, 949), (34, 1418)]

# Triangulate points
points_3D = triangulate(
    left_camera_mtx, right_camera_mtx, R, T, points_left, points_right
)

# Plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], c="r", s=50)

# Draw lines connecting the points
for i in range(len(points_3D) - 1):
    ax.plot(
        [points_3D[i, 0], points_3D[i + 1, 0]],
        [points_3D[i, 1], points_3D[i + 1, 1]],
        [points_3D[i, 2], points_3D[i + 1, 2]],
        c="r",
    )

plt.show()
