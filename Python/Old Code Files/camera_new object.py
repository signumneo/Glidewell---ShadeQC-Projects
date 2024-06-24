import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load rectification parameters
left_camera_mtx = np.load("camera_mtx_left.npy")
left_camera_dist = np.load("camera_dist_left.npy")
right_camera_mtx = np.load("camera_mtx_right.npy")
right_camera_dist = np.load("camera_dist_right.npy")
R1 = np.load("R1.npy")
R2 = np.load("R2.npy")
P1 = np.load("P1.npy")
P2 = np.load("P2.npy")
Q = np.load("Q.npy")


# Capture new stereo images
left_img_path = r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_left_1.jpg"
right_img_path = r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_right_1.jpg"

# Load the new images
left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Left Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Left Image", 800, 800)
cv2.imshow("Left Image", left_img)

cv2.namedWindow("Right Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right Image", 800, 800)
cv2.imshow("Right Image", right_img)

# Compute rectification maps
left_map_x, left_map_y = cv2.initUndistortRectifyMap(
    left_camera_mtx, left_camera_dist, R1, P1, left_img.shape[::1], cv2.CV_32FC1
)
right_map_x, right_map_y = cv2.initUndistortRectifyMap(
    right_camera_mtx, right_camera_dist, R2, P2, right_img.shape[::1], cv2.CV_32FC1
)

# Apply rectification maps
left_rectified = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

# Display the cropped rectified images
cv2.namedWindow("Left Rectified Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Left Rectified Image", 800, 800)
cv2.imshow("Left Rectified Image", left_rectified)

cv2.namedWindow("Right Rectified Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right Rectified Image", 800, 800)
cv2.imshow("Right Rectified Image", right_rectified)

# Calculate ROI to remove black borders
_, roi_left = cv2.getOptimalNewCameraMatrix(
    left_camera_mtx, left_camera_dist, left_img.shape[::1], 2, left_img.shape[::1]
)
_, roi_right = cv2.getOptimalNewCameraMatrix(
    right_camera_mtx, right_camera_dist, right_img.shape[::1], 2, right_img.shape[::1]
)

# Crop the rectified images using the ROI
x, y, w, h = roi_left
left_rectified_cropped = left_rectified[y : y + h, x : x + w]

x, y, w, h = roi_right
right_rectified_cropped = right_rectified[y : y + h, x : x + w]

# Display the cropped rectified images
cv2.namedWindow("Cropped Left Rectified Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cropped Left Rectified Image", 800, 800)
cv2.imshow("Cropped Left Rectified Image", left_rectified_cropped)

cv2.namedWindow("Cropped Right Rectified Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cropped Right Rectified Image", 800, 800)
cv2.imshow("Cropped Right Rectified Image", right_rectified_cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute the width for side-by-side stitching without black borders
height, width = right_rectified.shape

# Resize rectified images to the same height while maintaining the aspect ratio
left_rectified_resized = cv2.resize(
    left_rectified, (width, height), interpolation=cv2.INTER_LINEAR
)
right_rectified_resized = cv2.resize(
    right_rectified, (width, height), interpolation=cv2.INTER_LINEAR
)

# Detect ORB keypoints and descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(left_rectified_resized, None)
keypoints2, descriptors2 = orb.detectAndCompute(right_rectified_resized, None)

# Match descriptors using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance and select top 30 matches
matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:70]

# Extract matched keypoints
pts_left = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_right = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Stitch the images together side-by-side
stitched_img = np.hstack((left_rectified_resized, right_rectified_resized))

# Convert to color for visualization
stitched_img_color = cv2.cvtColor(stitched_img, cv2.COLOR_GRAY2BGR)

# Draw key epipolar lines
for pt1, pt2 in zip(pts_left, pts_right):
    color = tuple(np.random.randint(0, 255, 3).tolist())
    pt1 = (int(pt1[0][0]), int(pt1[0][1]))
    pt2 = (
        int(pt2[0][0] + width),
        int(pt2[0][1]),
    )  # Shift the right point horizontally by the width of the left image
    stitched_img_color = cv2.line(stitched_img_color, pt1, pt2, color, 1)
    stitched_img_color = cv2.circle(stitched_img_color, pt1, 8, color, -1)
    stitched_img_color = cv2.circle(stitched_img_color, pt2, 8, color, -1)

# Display the stitched image with epipolar lines
cv2.namedWindow("Epipolar Lines", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Epipolar Lines", 1600, 600)
cv2.imshow("Epipolar Lines", stitched_img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the stitched image with epipolar lines if needed
cv2.imwrite("stitched_epipolar_lines_new_object.jpg", stitched_img_color)

# Triangulate points
points_4D_hom = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
points_3D = points_4D_hom[:3] / points_4D_hom[3]

# Plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points_3D[0], points_3D[1], points_3D[2], c="r", s=50)

# Draw lines connecting the points
for i in range(len(points_3D) - 1):
    ax.plot(
        [points_3D[i, 0], points_3D[i + 1, 0]],
        [points_3D[i, 1], points_3D[i + 1, 1]],
        [points_3D[i, 2], points_3D[i + 1, 2]],
        c="r",
    )

plt.show()
