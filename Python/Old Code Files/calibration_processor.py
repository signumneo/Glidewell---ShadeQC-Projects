import os
import glob
import cv2
import numpy as np
from calibration_data import CalibrationData


class CalibrationProcessor:
    def __init__(self, left_folder, right_folder, chessboard_size, square_size):
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.chessboard_size = chessboard_size
        self.square_size = square_size

    def calibrate_individual_camera(self, images_folder, camera_name):
        images = glob.glob(os.path.join(images_folder, "*.jpg"))

        # Prepare object points
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)
        objp *= self.square_size

        # Arrays to store object points and image points from all the images
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane

        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            if ret:
                img_points.append(corners)
                obj_points.append(objp)

        if len(img_points) == 0:
            print(
                f"No chessboard corners found in any images for {camera_name} camera."
            )
            return

        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        calibration_data = CalibrationData(mtx, dist, rvecs, tvecs)
        calibration_data.save(f"calibration_data_{camera_name}.json")

        print(f"{camera_name} camera calibrated successfully and data saved to JSON.")

    def calibrate_cameras(self):
        self.calibrate_individual_camera(self.left_folder, "left")
        self.calibrate_individual_camera(self.right_folder, "right")
