import numpy as np

# Load intrinsic parameters
camera_mtx_left = np.load("camera_mtx_left.npy")
dist_left = np.load("camera_dist_left.npy")
camera_mtx_right = np.load("camera_mtx_right.npy")
dist_right = np.load("camera_dist_right.npy")

# Load extrinsic parameters
R = np.load("R.npy")
T = np.load("T.npy")

# Load rectification parameters
R1 = np.load("R1.npy")
R2 = np.load("R2.npy")
P1 = np.load("P1.npy")
P2 = np.load("P2.npy")
Q = np.load("Q.npy")

# Print the loaded parameters for verification
print("Camera Matrix Left:\n", camera_mtx_left)
print("Distortion Coefficients Left:\n", dist_left)
print("Camera Matrix Right:\n", camera_mtx_right)
print("Distortion Coefficients Right:\n", dist_right)
print("Rotation Matrix:\n", R)
print("Translation Vector:\n", T)
print("Rectification Transform R1:\n", R1)
print("Rectification Transform R2:\n", R2)
print("Projection Matrix P1:\n", P1)
print("Projection Matrix P2:\n", P2)
print("Disparity-to-depth Mapping Matrix Q:\n", Q)
