import cv2
import numpy as np
import glob

# Define the chessboard size (number of inner corners per row and column)
chessboard_size = (8, 11)

# Define the square size in meters (5 mm = 0.005 meters)
square_size = 0.005

# Define the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (7,10,0) scaled by the square size
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Load images from the specified folder
images = glob.glob(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\RightCamera\*.jpg"
)

img = None

for fname in images:
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_display = cv2.drawChessboardCorners(
            img_display, chessboard_size, corners2, ret
        )
        img_display_resized = cv2.resize(img_display, (700, 800))
        cv2.imshow("Chessboard Corners", img_display_resized)
        cv2.waitKey(5000)  # Adjust display time as needed

cv2.destroyAllWindows()

# Check if img is not None
if img is not None:
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[::-1], None, None
    )

    # Calculate and display the reprojection error
    mean_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error * len(imgpoints2)
        total_points += len(imgpoints2)

    mean_error /= total_points

    # Print the calibration results and reprojection error
    print("Camera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)
    print("\nMean reprojection error:")
    print(mean_error)

    # Save the camera calibration results
    np.save("camera_mtx_right.npy", mtx)
    np.save("camera_dist_right.npy", dist)

    # Test undistortion on an image
    img = cv2.imread(images[1], cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Resize for display
    img_resized = cv2.resize(img, (700, 800))
    dst_resized = cv2.resize(dst, (700, 800))

    cv2.imshow("Original Image", img_resized)
    cv2.imshow("Calibrated Image", dst_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No chessboard corners found in any images. Calibration failed.")
