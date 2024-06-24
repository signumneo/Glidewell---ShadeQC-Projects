import cv2
import numpy as np

# Load the saved calibration results
left_camera_mtx = np.load("camera_mtx_left.npy")
left_camera_dist = np.load("dist_left.npy")
right_camera_mtx = np.load("camera_mtx_right.npy")
right_camera_dist = np.load("dist_right.npy")
R = np.load("R.npy")
T = np.load("T.npy")

square_size = 0.005

# Paths to the checkerboard images used for stereo calibration
left_img_path = r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_left_1.jpg"
right_img_path = r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_right_1.jpg"

# Load the checkerboard images
left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

# Compute rectification transforms
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    left_camera_mtx,
    left_camera_dist,
    right_camera_mtx,
    right_camera_dist,
    left_img.shape[::-1],
    R,
    T,
)

# Save the rectification parameters
np.save("R1.npy", R1)
np.save("R2.npy", R2)
np.save("P1.npy", P1)
np.save("P2.npy", P2)
np.save("Q.npy", Q)

# Compute the rectification maps
left_map_x, left_map_y = cv2.initUndistortRectifyMap(
    left_camera_mtx, left_camera_dist, R1, P1, left_img.shape[::-1], cv2.CV_32FC1
)
right_map_x, right_map_y = cv2.initUndistortRectifyMap(
    right_camera_mtx, right_camera_dist, R2, P2, right_img.shape[::-1], cv2.CV_32FC1
)

# Apply the rectification maps
left_rectified = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

# Crop the rectified images using the ROI
x, y, w, h = roi1
left_rectified_cropped = left_rectified[y : y + h, x : x + w]
x, y, w, h = roi2
right_rectified_cropped = right_rectified[y : y + h, x : x + w]

# Resize rectified images to a fixed size (e.g., 2800x1000)
desired_size = (2800, 1000)
left_rectified_resized = cv2.resize(
    left_rectified_cropped, desired_size, interpolation=cv2.INTER_LINEAR
)
right_rectified_resized = cv2.resize(
    right_rectified_cropped, desired_size, interpolation=cv2.INTER_LINEAR
)

# Display cropped rectified images
cv2.namedWindow("Left Rectified", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Left Rectified", 800, 1200)
cv2.imshow("Left Rectified", left_rectified_resized)
cv2.namedWindow("Right Rectified", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right Rectified", 800, 1200)
cv2.imshow("Right Rectified", right_rectified_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save rectified images if needed
cv2.imwrite("left_rectified_cropped.jpg", left_rectified_resized)
cv2.imwrite("right_rectified_cropped.jpg", right_rectified_resized)


# Calculate reprojection error
def calculate_reprojection_error(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    rvecs,
    tvecs,
    mtx_left,
    dist_left,
    mtx_right,
    dist_right,
):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2_left, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx_left, dist_left
        )
        imgpoints2_right, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx_right, dist_right
        )
        error_left = cv2.norm(imgpoints_left[i], imgpoints2_left, cv2.NORM_L2) / len(
            imgpoints2_left
        )
        error_right = cv2.norm(imgpoints_right[i], imgpoints2_right, cv2.NORM_L2) / len(
            imgpoints2_right
        )
        total_error += error_left + error_right
        total_points += len(imgpoints2_left) + len(imgpoints2_right)
    mean_error = total_error / total_points
    return mean_error


"""# Assuming objpoints, imgpoints_left, and imgpoints_right are available from stereo calibration
# The objpoints should be generated and imgpoints detected during stereo calibration process

# Example objpoints and imgpoints (these should be computed during your calibration process)
objpoints = [
    np.array([[x, y, 0] for y in range(8) for x in range(11)], dtype=np.float32)
    * square_size
] * 10  # Example, replace with actual
imgpoints_left = [corners_left] * 10  # Example, replace with actual detected corners
imgpoints_right = [corners_right] * 10  # Example, replace with actual detected corners"""


# Stereo calibration to get rvecs and tvecs
_, rvecs, tvecs = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    left_camera_mtx,
    left_camera_dist,
    right_camera_mtx,
    right_camera_dist,
    left_img.shape[::-1],
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC,
)

# Calculate mean reprojection error
mean_error = calculate_reprojection_error(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    rvecs,
    tvecs,
    left_camera_mtx,
    left_camera_dist,
    right_camera_mtx,
    right_camera_dist,
)
print("Mean reprojection error (stereo):", mean_error)
