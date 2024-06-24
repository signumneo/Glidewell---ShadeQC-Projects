import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load rectified images
left_img = cv2.imread(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\left_rectified_cropped.jpg",
    cv2.IMREAD_GRAYSCALE,
)
right_img = cv2.imread(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\right_rectified_cropped.jpg",
    cv2.IMREAD_GRAYSCALE,
)

# Detect ORB keypoints and descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(left_img, None)
keypoints2, descriptors2 = orb.detectAndCompute(right_img, None)

# Match descriptors using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints
pts_left = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_right = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Load the fundamental matrix
F = np.load("F.npy")

# Stitch the images together side-by-side
height, width = left_img.shape
stitched_img = np.hstack((left_img, right_img))

# Convert to color for visualization
stitched_img_color = cv2.cvtColor(stitched_img, cv2.COLOR_GRAY2BGR)

# Draw epipolar lines
for pt1, pt2 in zip(pts_left, pts_right):
    color = tuple(np.random.randint(0, 255, 3).tolist())
    pt1 = (int(pt1[0][0]), int(pt1[0][1]))
    pt2 = (
        int(pt2[0][0] + width),
        int(pt2[0][1]),
    )  # Shift the right point horizontally by the width of the left image
    stitched_img_color = cv2.line(stitched_img_color, pt1, pt2, color, 1)
    stitched_img_color = cv2.circle(stitched_img_color, pt1, 5, color, -1)
    stitched_img_color = cv2.circle(stitched_img_color, pt2, 5, color, -1)

# Display the stitched image with epipolar lines
cv2.namedWindow("Epipolar Lines", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Epipolar Lines", 1600, 600)
cv2.imshow("Epipolar Lines", stitched_img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the stitched image with epipolar lines if needed
cv2.imwrite("stitched_epipolar_lines.jpg", stitched_img_color)
