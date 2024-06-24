import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_calibration_data(filename):
    npzfile = np.load(filename)
    mtx = npzfile["mtx"]
    dist = npzfile["dist"]
    rvecs = npzfile["rvecs"]
    tvecs = npzfile["tvecs"]
    objpoints = npzfile["objpoints"]
    imgpoints = npzfile["imgpoints"]
    return mtx, dist, rvecs, tvecs, objpoints, imgpoints


def collect_stereo_image_points(pairs_folder, rows=11, columns=8, world_scaling=1.0):
    left_images = sorted(glob.glob(pairs_folder + "/pair_left_*.png"))
    right_images = sorted(glob.glob(pairs_folder + "/pair_right_*.png"))

    assert len(left_images) == len(
        right_images
    ), "The number of left and right images must be the same."

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    objp = world_scaling * objp

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    for left_img_name, right_img_name in zip(left_images, right_images):
        left_img = cv.imread(left_img_name)
        right_img = cv.imread(right_img_name)

        if left_img is None or right_img is None:
            print(f"Error loading images: {left_img_name}, {right_img_name}")
            continue

        gray_left = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

        ret_left, corners_left = cv.findChessboardCorners(
            gray_left, (columns, rows), None
        )
        ret_right, corners_right = cv.findChessboardCorners(
            gray_right, (columns, rows), None
        )

        if ret_left and ret_right:
            corners_left = cv.cornerSubPix(
                gray_left, corners_left, (11, 11), (-1, -1), criteria
            )
            corners_right = cv.cornerSubPix(
                gray_right, corners_right, (11, 11), (-1, -1), criteria
            )

            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

    return objpoints, imgpoints_left, imgpoints_right, gray_left.shape[::-1]


def stereo_calibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    mtx_left,
    dist_left,
    mtx_right,
    dist_right,
    image_size,
):
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = (
        cv.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            mtx_left,
            dist_left,
            mtx_right,
            dist_right,
            image_size,
            criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            flags=cv.CALIB_FIX_INTRINSIC,
        )
    )
    return retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F


# Load individual calibration data
mtx_left, dist_left, rvecs_left, tvecs_left, objpoints_left, imgpoints_left = (
    load_calibration_data(
        "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\Latest Code Files\\left_camera_calibration.npz"
    )
)
mtx_right, dist_right, rvecs_right, tvecs_right, objpoints_right, imgpoints_right = (
    load_calibration_data(
        "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\Latest Code Files\\right_camera_calibration.npz"
    )
)

# Collect object points and image points for stereo calibration
objpoints, imgpoints_left, imgpoints_right, image_size = collect_stereo_image_points(
    "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\CapturedImages\\Pairs"
)

# Perform stereo calibration
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = (
    stereo_calibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
        image_size,
    )
)

# Print stereo calibration results
print(f"Stereo Calibration Results:")
print(f"Reprojection Error: {retval}")
print(f"Rotation Matrix:\n{R}")
print(f"Translation Vector:\n{T}")
print(f"Essential Matrix:\n{E}")
print(f"Fundamental Matrix:\n{F}")

# Save stereo calibration results
np.savez(
    "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\Latest Code Files\\stereo_calibration.npz",
    R=R,
    T=T,
    E=E,
    F=F,
    cameraMatrix1=cameraMatrix1,
    distCoeffs1=distCoeffs1,
    cameraMatrix2=cameraMatrix2,
    distCoeffs2=distCoeffs2,
)

# Load stereo calibration results
stereo_calibration_data = np.load(
    "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\Latest Code Files\\stereo_calibration.npz"
)
R = stereo_calibration_data["R"]
T = stereo_calibration_data["T"]
cameraMatrix1 = stereo_calibration_data["cameraMatrix1"]
distCoeffs1 = stereo_calibration_data["distCoeffs1"]
cameraMatrix2 = stereo_calibration_data["cameraMatrix2"]
distCoeffs2 = stereo_calibration_data["distCoeffs2"]

# Perform stereo rectification with adjusted new camera matrix
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
    cameraMatrix1,
    distCoeffs1,
    cameraMatrix2,
    distCoeffs2,
    image_size,
    R,
    T,
    flags=cv.CALIB_ZERO_DISPARITY,
    alpha=1,
)

# Compute rectification maps
left_map1, left_map2 = cv.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, image_size, cv.CV_16SC2
)
right_map1, right_map2 = cv.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2, image_size, cv.CV_16SC2
)

# Example of loading a pair of stereo images
left_img = cv.imread(
    "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\CapturedImages\\Pairs\\pair_left_8.png"
)
right_img = cv.imread(
    "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\CapturedImages\\Pairs\\pair_right_8.png"
)

# Check if images are loaded correctly
if left_img is None:
    print("Error: Unable to load left image.")
if right_img is None:
    print("Error: Unable to load right image.")

# Rectify the images
if left_img is not None and right_img is not None:
    left_rectified = cv.remap(left_img, left_map1, left_map2, cv.INTER_LINEAR)
    right_rectified = cv.remap(right_img, right_map1, right_map2, cv.INTER_LINEAR)

    # Resize output window for better visualization
    resized_left_rectified = cv.resize(left_rectified, (800, 900))
    resized_right_rectified = cv.resize(right_rectified, (800, 900))

    cv.imshow("Left Rectified", resized_left_rectified)
    cv.imshow("Right Rectified", resized_right_rectified)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Compute disparity map
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=20)
    disparity = stereo.compute(
        cv.cvtColor(left_rectified, cv.COLOR_BGR2GRAY),
        cv.cvtColor(right_rectified, cv.COLOR_BGR2GRAY),
    )
    print(np.unique(disparity))

    # Reproject to 3D
    points_3D = cv.reprojectImageTo3D(disparity, Q)

    # Visualize the 3D points using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Mask the disparity map to ignore points with no disparity
    mask = disparity > disparity.min()
    points = points_3D[mask]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=disparity[mask], cmap="jet")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

    # Save the rectified images and disparity map if needed
    cv.imwrite("left_rectified.png", left_rectified)
    cv.imwrite("right_rectified.png", right_rectified)
    cv.imwrite("disparity.png", disparity)
