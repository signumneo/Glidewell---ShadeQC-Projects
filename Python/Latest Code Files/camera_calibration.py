import cv2 as cv
import glob
import numpy as np
import os


def save_calibration_data(filename, mtx, dist, rvecs, tvecs, objpoints, imgpoints):
    np.savez(
        filename,
        mtx=mtx,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        objpoints=objpoints,
        imgpoints=imgpoints,
    )


def load_calibration_data(filename):
    npzfile = np.load(filename)
    mtx = npzfile["mtx"]
    dist = npzfile["dist"]
    rvecs = npzfile["rvecs"]
    tvecs = npzfile["tvecs"]
    objpoints = npzfile["objpoints"]
    imgpoints = npzfile["imgpoints"]
    return mtx, dist, rvecs, tvecs, objpoints, imgpoints


def calibrate_camera(images_folder, rows=11, columns=8, world_scaling=1.0):
    images_names = sorted(glob.glob(images_folder))
    images = [cv.imread(imname, 1) for imname in images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    objp = world_scaling * objp

    objpoints = []
    imgpoints = []

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (columns, rows), None)

        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (columns, rows), corners, ret)

            # Resize the frame before displaying it
            resized_frame = cv.resize(frame, (800, 900))  # Resize to 800x900
            cv.imshow("img", resized_frame)
            cv.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyAllWindows()
    print("Number of images used for calibration:", len(objpoints))

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (gray.shape[1], gray.shape[0]), None, None
    )
    print("RMSE:", ret)
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:", dist)
    # print("Rotation Vectors:\n", rvecs)
    # print("Translation Vectors:\n", tvecs)

    # Verify calibration by reprojection
    total_error = 0
    for i, frame in enumerate(images):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error

        # Draw detected corners (original points)
        cv.drawChessboardCorners(frame, (columns, rows), imgpoints[i], True)

        # Draw reprojected points
        for j in range(len(imgpoints2)):
            cv.circle(
                frame,
                (int(imgpoints2[j][0][0]), int(imgpoints2[j][0][1])),
                3,
                (0, 255, 0),
                -1,
            )

        # Display the image with both sets of points
        resized_frame = cv.resize(frame, (800, 900))
        cv.imshow("Reprojection", resized_frame)
        cv.waitKey(500)

    cv.destroyAllWindows()

    mean_error = total_error / len(images)
    print(f"Mean Reprojection Error: {mean_error}")

    return mtx, dist, ret, rvecs, tvecs, objpoints, imgpoints


# Calibrate left camera
(
    mtx_left,
    dist_left,
    ret_left,
    rvecs_left,
    tvecs_left,
    objpoints_left,
    imgpoints_left,
) = calibrate_camera(
    images_folder="C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\CapturedImages\\LeftCamera\\*.png"
)

save_calibration_data(
    "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\Latest Code Files\\left_camera_calibration.npz",
    mtx_left,
    dist_left,
    rvecs_left,
    tvecs_left,
    objpoints_left,
    imgpoints_left,
)

# Calibrate right camera
(
    mtx_right,
    dist_right,
    ret_right,
    rvecs_right,
    tvecs_right,
    objpoints_right,
    imgpoints_right,
) = calibrate_camera(
    images_folder="C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\CapturedImages\\RightCamera\\*.png"
)

save_calibration_data(
    "C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\Latest Code Files\\right_camera_calibration.npz",
    mtx_right,
    dist_right,
    rvecs_right,
    tvecs_right,
    objpoints_right,
    imgpoints_right,
)

# Load calibration data (example)
# mtx_left, dist_left, rvecs_left, tvecs_left, objpoints_left, imgpoints_left = load_calibration_data("C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\Latest Code Files\\left_camera_calibration.npz")
# mtx_right, dist_right, rvecs_right, tvecs_right, objpoints_right, imgpoints_right = load_calibration_data("C:\\Users\\jacob.thomas\\Desktop\\Jacob Thomas_118577\\Full Time 2024\\ShadeQC - Jacob\\IDS camera - GUI\\Python\\Latest Code Files\\right_camera_calibration.npz")
