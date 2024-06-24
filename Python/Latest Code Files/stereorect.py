import cv2 as cv
import glob
import numpy as np


# Load stereo calibration results
calibration_data = np.load(
    "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\Latest Code Files\\stereo_calibration.npz"
)
R = calibration_data["R"]
T = calibration_data["T"]
cameraMatrix1 = calibration_data["cameraMatrix1"]
distCoeffs1 = calibration_data["distCoeffs1"]
cameraMatrix2 = calibration_data["cameraMatrix2"]
distCoeffs2 = calibration_data["distCoeffs2"]

# Perform stereo rectification
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, R, T
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
    "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\CapturedImages\\Pairs\\pair_left_1.png"
)
right_img = cv.imread(
    "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\CapturedImages\\Pairs\\pair_right_1.png"
)

# Rectify the images
left_rectified = cv.remap(left_img, left_map1, left_map2, cv.INTER_LINEAR)
right_rectified = cv.remap(right_img, right_map1, right_map2, cv.INTER_LINEAR)

# Compute disparity map
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(
    cv.cvtColor(left_rectified, cv.COLOR_BGR2GRAY),
    cv.cvtColor(right_rectified, cv.COLOR_BGR2GRAY),
)

# Reproject to 3D
points_3D = cv.reprojectImageTo3D(disparity, Q)

# Visualize rectified images
cv.imshow("Left Rectified", left_rectified)
cv.imshow("Right Rectified", right_rectified)
cv.imshow("Disparity", disparity / 16.0)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the rectified images and disparity map if needed
cv.imwrite("left_rectified.png", left_rectified)
cv.imwrite("right_rectified.png", right_rectified)
cv.imwrite("disparity.png", disparity)
