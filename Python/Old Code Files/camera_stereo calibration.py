import cv2
import numpy as np
import glob

# Load the saved calibration results
left_camera_mtx = np.load("camera_mtx_left.npy")
left_camera_dist = np.load("camera_dist_left.npy")
right_camera_mtx = np.load("camera_mtx_right.npy")
right_camera_dist = np.load("camera_dist_right.npy")

# Load the paired images for stereo calibration
left_images = glob.glob(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_left_*.jpg"
)
right_images = glob.glob(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_right_*.jpg"
)

# Ensure the lists are paired correctly
num_pairs = min(len(left_images), len(right_images))
left_images = left_images[:num_pairs]
right_images = right_images[:num_pairs]

# Arrays to store object points and image points from all the images
objpoints = []
left_imgpoints = []
right_imgpoints = []

# Prepare object points (0,0,0), (1,0,0), ..., (7,10,0)
chessboard_size = (8, 11)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)

# Define the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Detect chessboard corners in paired images
for left_img_path, right_img_path in zip(left_images, right_images):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    ret_left, corners_left = cv2.findChessboardCorners(left_img, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(
        right_img, chessboard_size, None
    )

    if ret_left and ret_right:
        objpoints.append(objp)

        corners_left2 = cv2.cornerSubPix(
            left_img, corners_left, (11, 11), (-1, -1), criteria
        )
        corners_right2 = cv2.cornerSubPix(
            right_img, corners_right, (11, 11), (-1, -1), criteria
        )

        left_imgpoints.append(corners_left2)
        right_imgpoints.append(corners_right2)

# Stereo calibration
ret, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    left_imgpoints,
    right_imgpoints,
    left_camera_mtx,
    left_camera_dist,
    right_camera_mtx,
    right_camera_dist,
    left_img.shape[::-1],
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC,
)

# Save the stereo calibration results
np.save("R.npy", R)
np.save("T.npy", T)
np.save("E.npy", E)
np.save("F.npy", F)

print("Stereo Calibration Results:")
print("Rotation Matrix:\n", R)
print("Translation Vector:\n", T)
print("Essential Matrix:\n", E)
print("Fundamental Matrix:\n", F)
