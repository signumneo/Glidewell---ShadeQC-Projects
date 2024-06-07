import os
import cv2
import numpy as np
import json
from calibration_data import CalibrationData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CalibrationProcessor:
    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((8 * 11, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)  # 8x11 checkerboard

    def perform_individual_calibration(self):
        for camera in ["LeftCamera", "RightCamera"]:
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.

            camera_dir = os.path.join(
                os.path.dirname(__file__), "CapturedImages", camera
            )
            print(camera_dir)
            if not os.path.isdir(camera_dir):
                print(f"No directory found for {camera}.")
                continue

            images = [
                os.path.join(camera_dir, fname)
                for fname in os.listdir(camera_dir)
                if fname.endswith(".jpg")
            ]

            if not images:
                print(f"No images found in {camera} folder.")
                continue

            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
                if ret:
                    objpoints.append(self.objp)
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), self.criteria
                    )
                    imgpoints.append(corners2)

                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (8, 11), corners2, ret)
                    window_name = f"Corners for {camera}"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 600, 400)  # Resize the window
                    cv2.imshow(window_name, img)
                    cv2.waitKey(5000)  # Display each image for 500 ms

            cv2.destroyAllWindows()  # Close all OpenCV windows

            if not imgpoints:
                print(f"No valid checkerboard patterns found in {camera} images.")
                continue

            ret, mtx, dist, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            # Format the calibration data to 2 decimal places
            mtx = np.round(mtx, 2)
            dist = np.round(dist, 2)

            calibration_data = {
                "camera_matrix": mtx.tolist(),
                "distortion_coefficients": dist.tolist(),
            }

            # Saving the calibration data
            file_path = os.path.join(
                os.path.dirname(__file__), f"calibration_data_{camera.lower()}.json"
            )
            with open(file_path, "w") as f:
                json.dump(calibration_data, f)
            print(f"Calibration for {camera} successful. Data saved to {file_path}.")

    def perform_stereo_calibration(self):
        objpoints = []  # 3d point in real world space
        imgpoints_left = []  # 2d points in image plane for left camera
        imgpoints_right = []  # 2d points in image plane for right camera

        pairs_dir = os.path.join(os.path.dirname(__file__), "CapturedImages", "Pairs")
        if not os.path.isdir(pairs_dir):
            print("No directory found for Pairs.")
            return

        pairs = [
            os.path.join(pairs_dir, fname)
            for fname in os.listdir(pairs_dir)
            if fname.endswith(".jpg")
        ]

        if not pairs:
            print("No images found in Pairs folder.")
            return

        for i in range(0, len(pairs), 2):
            if i + 1 >= len(pairs):
                break

            img_left = cv2.imread(pairs[i])
            img_right = cv2.imread(pairs[i])
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            ret_left, corners_left = cv2.findChessboardCorners(gray_left, (8, 11), None)
            ret_right, corners_right = cv2.findChessboardCorners(
                gray_right, (8, 11), None
            )

            if ret_left and ret_right:
                objpoints.append(self.objp)
                corners2_left = cv2.cornerSubPix(
                    gray_left, corners_left, (11, 11), (-1, -1), self.criteria
                )
                imgpoints_left.append(corners2_left)
                corners2_right = cv2.cornerSubPix(
                    gray_right, corners_right, (11, 11), (-1, -1), self.criteria
                )
                imgpoints_right.append(corners2_right)

        if not imgpoints_left or not imgpoints_right:
            print("No valid checkerboard patterns found in pairs.")
            return

        # Load individual calibration data
        base_dir = os.path.dirname(__file__)
        left_calib_path = os.path.join(base_dir, "calibration_data_leftcamera.json")
        right_calib_path = os.path.join(base_dir, "calibration_data_rightcamera.json")

        calibration_data_left = CalibrationData.load(left_calib_path)
        calibration_data_right = CalibrationData.load(right_calib_path)

        if not calibration_data_left or not calibration_data_right:
            print("Individual calibration data not found.")
            return

        # Stereo calibration
        (
            ret,
            mtxL,
            distL,
            mtxR,
            distR,
            R,
            T,
            E,
            F,
        ) = cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            calibration_data_left.camera_matrix,
            calibration_data_left.distortion_coefficients,
            calibration_data_right.camera_matrix,
            calibration_data_right.distortion_coefficients,
            gray_left.shape[::-1],
            criteria=self.criteria,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )

        # Format the calibration data to 2 decimal places
        R = np.round(R, 2)
        T = np.round(T, 2)
        E = np.round(E, 2)
        F = np.round(F, 2)

        calibration_data_stereo = {
            "rotation_matrix": R.tolist(),
            "translation_vector": T.tolist(),
            "essential_matrix": E.tolist(),
            "fundamental_matrix": F.tolist(),
        }

        stereo_calib_path = os.path.join(base_dir, "calibration_data_stereo.json")
        with open(stereo_calib_path, "w") as f:
            json.dump(calibration_data_stereo, f)
        print(
            "Stereo calibration successful. Data saved to calibration_data_stereo.json."
        )

        # Function to load the calibration data from a JSON file

        # Functions for stereo rectification, disparity map calculation and depth map estimation

    def load_calibration_data(self, file_path):
        if not os.path.exists(file_path):
            print(f"Calibration data file not found: {file_path}")
            return None
        with open(file_path, "r") as f:
            return json.load(f)

    def stereo_rectify(self):
        # Load calibration data
        base_dir = os.path.dirname(__file__)
        left_calib = self.load_calibration_data(
            os.path.join(base_dir, "calibration_data_leftcamera.json")
        )
        right_calib = self.load_calibration_data(
            os.path.join(base_dir, "calibration_data_rightcamera.json")
        )
        stereo_calib = self.load_calibration_data(
            os.path.join(base_dir, "calibration_data_stereo.json")
        )

        mtxL = np.array(left_calib["camera_matrix"])
        distL = np.array(left_calib["distortion_coefficients"])
        mtxR = np.array(right_calib["camera_matrix"])
        distR = np.array(right_calib["distortion_coefficients"])
        R = np.array(stereo_calib["rotation_matrix"])
        T = np.array(stereo_calib["translation_vector"])

        # Round all values to 2 decimal places
        mtxL = np.round(mtxL, 2)
        distL = np.round(distL, 2)
        mtxR = np.round(mtxR, 2)
        distR = np.round(distR, 2)
        R = np.round(R, 2)
        T = np.round(T, 2)

        print("Left Camera Matrix:", mtxL)
        print("Left Distortion Coefficients:", distL)
        print("Right Camera Matrix:", mtxR)
        print("Right Distortion Coefficients:", distR)
        print("Rotation Matrix:", R)
        print("Translation Vector:", T)

        # Load example images to detect the image size
        left_img = cv2.imread(
            r"Camera - GUI\CapturedImages\LeftCamera\left_image_1.jpg",
            cv2.IMREAD_GRAYSCALE,
        )
        right_img = cv2.imread(
            r"Camera - GUI\CapturedImages\RightCamera\right_image_1.jpg",
            cv2.IMREAD_GRAYSCALE,
        )

        if left_img is None or right_img is None:
            print("Failed to load images.")
            return

        image_size = (left_img.shape[1], left_img.shape[0])  # (width, height)

        # Get the optimal new camera matrix
        new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(
            mtxL, distL, image_size, 1, image_size
        )
        new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(
            mtxR, distR, image_size, 1, image_size
        )

        # Compute rectification transforms
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            new_mtxL, distL, new_mtxR, distR, image_size, R, T, alpha=0
        )

        # Round all rectification values to 2 decimal places
        R1 = np.round(R1, 2)
        R2 = np.round(R2, 2)
        P1 = np.round(P1, 2)
        P2 = np.round(P2, 2)
        Q = np.round(Q, 2)

        print("R1:", R1)
        print("R2:", R2)
        print("P1:", P1)
        print("P2:", P2)
        print("Q:", Q)

        # Compute rectification maps
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            mtxL, distL, None, new_mtxL, image_size, 5
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            mtxR, distR, None, new_mtxR, image_size, 5
        )

        # Print maps for debugging
        print("Left Map1:", left_map1)
        print("Left Map2:", left_map2)
        print("Right Map1:", right_map1)
        print("Right Map2:", right_map2)

        # Remap images
        left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

        # Save the rectified images
        cv2.imwrite("left_rectified.jpg", left_rectified)
        cv2.imwrite("right_rectified.jpg", right_rectified)

        # Display the original, undistorted, and rectified images
        cv2.namedWindow("Original Left Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original Left Image", 600, 800)
        cv2.imshow("Original Left Image", left_img)

        cv2.namedWindow("Original Right Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original Right Image", 600, 800)
        cv2.imshow("Original Right Image", right_img)

        cv2.namedWindow("Rectified Left Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Rectified Left Image", 600, 800)
        cv2.imshow("Rectified Left Image", left_rectified)

        cv2.namedWindow("Rectified Right Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Rectified Right Image", 600, 800)
        cv2.imshow("Rectified Right Image", right_rectified)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return left_rectified, right_rectified, Q

    def visualize_disparity(self, left_rectified, right_rectified):
        # Compute disparity map with adjusted parameters
        numDisparities = 16 * 5  # Should be divisible by 16
        blockSize = 15  # Must be odd
        stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
        disparity = stereo.compute(left_rectified, right_rectified)

        # Print minimum and maximum disparity values
        min_disp = disparity.min()
        max_disp = disparity.max()
        print(f"Disparity min: {min_disp}, max: {max_disp}")

        # Normalize the disparity map for visualization
        if min_disp == max_disp:
            print("Disparity values are constant, likely an error in computation.")
            return

        disparity_normalized = cv2.normalize(
            disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        disparity_normalized = np.uint8(disparity_normalized)

        # Save and display disparity map
        base_dir = os.path.dirname(__file__)
        cv2.imwrite(os.path.join(base_dir, "disparity_map.jpg"), disparity_normalized)
        cv2.namedWindow("Disparity Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Disparity Map", 600, 800)
        cv2.imshow("Disparity Map", disparity_normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return disparity

    def visualize_3d_points(self, disparity, Q):
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        # Load the rectified left image to extract colors
        left_rectified = cv2.imread(
            os.path.join(os.path.dirname(__file__), "left_rectified.jpg"),
            cv2.IMREAD_GRAYSCALE,
        )

        mask_map = disparity > disparity.min()

        output_points = points_3D[mask_map]
        output_colors = left_rectified[mask_map]

        # Normalize colors for visualization
        output_colors = output_colors / 255.0

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            output_points[:, 0],
            output_points[:, 1],
            output_points[:, 2],
            c=output_colors / 255.0,
            s=1,
        )
        plt.show()

        return output_points, output_colors
    
    


if __name__ == "__main__":
    cp = CalibrationProcessor()
    cp.perform_individual_calibration()
    cp.perform_stereo_calibration()
    left_rectified, right_rectified, Q = cp.stereo_rectify()
    disparity = cp.visualize_disparity(left_rectified, right_rectified)
    cp.visualize_3d_points(disparity, Q)
