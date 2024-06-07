import cv2
import numpy as np
import json
import os


def load_calibration_data(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def rectify_image(image_path, calibration_data_path):
    # Load calibration data
    calibration_data = load_calibration_data(calibration_data_path)
    camera_matrix = np.array(calibration_data["camera_matrix"])
    dist_coeffs = np.array(calibration_data["distortion_coefficients"])

    print("Camera Matrix:", camera_matrix)
    print("Distortion Coefficients:", dist_coeffs)

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image at {image_path}")
        return

    # Get image size
    h, w = img.shape[:2]

    # Get the optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    print("New Camera Matrix:", new_camera_matrix)
    print("ROI:", roi)

    # Undistort the image alone for comparison
    undistorted_img = cv2.undistort(
        img, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    # Undistort and rectify the image
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5
    )
    rectified_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    # Display the original, undistorted, and rectified images
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Image", 600, 800)
    cv2.imshow("Original Image", img)

    cv2.namedWindow("Undistorted Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Undistorted Image", 600, 800)
    cv2.imshow("Undistorted Image", undistorted_img)

    cv2.namedWindow("Rectified Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rectified Image", 600, 800)
    cv2.imshow("Rectified Image", rectified_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Path to the calibration data and image
    calibration_data_path = "Camera - GUI\calibration_data_leftcamera.json"
    image_path = "Camera - GUI\CapturedImages\LeftCamera\left_image_1.jpg"

    rectify_image(image_path, calibration_data_path)
