import cv2
import numpy as np
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (8, 11)
SQUARE_SIZE = 5  # square size in mm

# Termination criteria for subpixel corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the real-world coordinates of the checkerboard corners
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[1], 0 : CHECKERBOARD[0]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real-world space
imgpoints_l = []  # 2d points in left image plane
imgpoints_r = []  # 2d points in right image plane

# Load images for the left and right cameras
images_left = glob.glob(
    r"C:\Users\jacob\OneDrive\Desktop\Jacob_Personal Projects\SLS\Captured Frames\Calibration\Left\left_*.jpg"
)
images_right = glob.glob(
    r"C:\Users\jacob\OneDrive\Desktop\Jacob_Personal Projects\SLS\Captured Frames\Calibration\Right\right_*.jpg"
)

# Ensure equal number of images are loaded
if len(images_left) != len(images_right):
    print("Mismatch in the number of left and right images.")
    print(f"Left images: {len(images_left)}, Right images: {len(images_right)}")
    exit()

print(f"Number of image pairs loaded: {len(images_left)}")

for i, (img_left, img_right) in enumerate(zip(images_left, images_right)):
    print(f"Processing image pair {i + 1}:")
    print(f"Left image path: {img_left}")
    print(f"Right image path: {img_right}")

    # Read the images
    img_l = cv2.imread(img_left)
    img_r = cv2.imread(img_right)

    # Resize images to 800x900
    img_l = cv2.resize(img_l, (800, 900))
    img_r = cv2.resize(img_r, (800, 900))

    # Convert the images to grayscale for corner detection
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners in the grayscale images
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

    # Check if the corners were found
    if not ret_l:
        print(f"Checkerboard corners not found in left image {img_left}")
    if not ret_r:
        print(f"Checkerboard corners not found in right image {img_right}")

    # If found, refine the corners and add them to the list
    if ret_l and ret_r:
        objpoints.append(objp)

        corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        imgpoints_l.append(corners2_l)

        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        imgpoints_r.append(corners2_r)

        # Draw and display the corners on the resized color images
        cv2.drawChessboardCorners(img_l, CHECKERBOARD, corners2_l, ret_l)
        cv2.drawChessboardCorners(img_r, CHECKERBOARD, corners2_r, ret_r)

        # Combine the images side by side
        combined_image = cv2.hconcat([img_l, img_r])

        # Display the combined image
        cv2.imshow("Combined Image with Corners", combined_image)
        cv2.waitKey(500)
    else:
        print(f"Skipping image pair {i + 1} due to missing corners.")

cv2.destroyAllWindows()

# Ensure at least one valid pair was found before calibration
if len(objpoints) == 0 or len(imgpoints_l) == 0 or len(imgpoints_r) == 0:
    print("No valid image pairs found for calibration.")
    exit()

# Perform calibration for the left camera
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    objpoints, imgpoints_l, gray_l.shape[::-1], None, None
)

# Perform calibration for the right camera
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    objpoints, imgpoints_r, gray_r.shape[::-1], None, None
)

# Calculate reprojection error for left camera
total_error_l = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs_l[i], tvecs_l[i], mtx_l, dist_l
    )
    error = cv2.norm(imgpoints_l[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error_l += error
mean_error_l = total_error_l / len(objpoints)

# Calculate reprojection error for right camera
total_error_r = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r
    )
    error = cv2.norm(imgpoints_r[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error_r += error
mean_error_r = total_error_r / len(objpoints)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC
ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_l,
    imgpoints_r,
    mtx_l,
    dist_l,
    mtx_r,
    dist_r,
    gray_l.shape[::-1],
    criteria=criteria,
    flags=flags,
)

# Stereo rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1], R, T
)

# Print out all parameters

print("\n--- Calibration Parameters ---\n")

# Intrinsic Parameters
print("Intrinsic Matrix for Left Camera (mtx_l):\n", mtx_l)
print("Distortion Coefficients for Left Camera (dist_l):\n", dist_l)
print("Intrinsic Matrix for Right Camera (mtx_r):\n", mtx_r)
print("Distortion Coefficients for Right Camera (dist_r):\n", dist_r)

# Extrinsic Parameters
print("\n--- Extrinsic Parameters ---\n")
print("Rotation Matrix between Cameras (R):\n", R)
print("Translation Vector between Cameras (T):\n", T)

# Reprojection Errors
print("\n--- Reprojection Errors ---\n")
print(f"Mean Reprojection Error for Left Camera: {mean_error_l}")
print(f"Mean Reprojection Error for Right Camera: {mean_error_r}")

# Stereo Calibration Parameters
print("\n--- Stereo Calibration Parameters ---\n")
print("Essential Matrix (E):\n", E)
print("Fundamental Matrix (F):\n", F)

# Stereo Rectification Parameters
print("\n--- Stereo Rectification Parameters ---\n")
print("Rectification Transform for Left Camera (R1):\n", R1)
print("Rectification Transform for Right Camera (R2):\n", R2)
print("Projection Matrix for Left Camera after Rectification (P1):\n", P1)
print("Projection Matrix for Right Camera after Rectification (P2):\n", P2)
print("Disparity-to-Depth Mapping Matrix (Q):\n", Q)

# Save calibration data
calibration_data = {
    "mtx_l": mtx_l,
    "dist_l": dist_l,
    "mtx_r": mtx_r,
    "dist_r": dist_r,
    "R": R,
    "T": T,
    "R1": R1,
    "R2": R2,
    "P1": P1,
    "P2": P2,
    "Q": Q,
}

np.save("calibration_data.npy", calibration_data)

print("\nCalibration completed and saved.")
