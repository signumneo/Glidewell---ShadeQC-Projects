import cv2
import numpy as np

# Initialize lists to store points for both images
points_left = []
points_right = []


# Mouse callback function for left image
def select_point_left(event, x, y, flags, param):
    global points_left, img_left
    if event == cv2.EVENT_LBUTTONDOWN:
        points_left.append((x, y))
        cv2.circle(img_left, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Left Image", img_left)
        print(f"Left Image Point selected: ({x}, {y})")


# Mouse callback function for right image
def select_point_right(event, x, y, flags, param):
    global points_right, img_right
    if event == cv2.EVENT_LBUTTONDOWN:
        points_right.append((x, y))
        cv2.circle(img_right, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Right Image", img_right)
        print(f"Right Image Point selected: ({x}, {y})")


# Load the images
img_left = cv2.imread(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_left_1.jpg"
)
img_right = cv2.imread(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_right_1.jpg"
)

# Display the images and set the mouse callbacks
cv2.namedWindow("Left Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Left Image", 600, 800)
cv2.imshow("Left Image", img_left)
cv2.setMouseCallback("Left Image", select_point_left)

cv2.namedWindow("Right Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right Image", 600, 800)
cv2.imshow("Right Image", img_right)
cv2.setMouseCallback("Right Image", select_point_right)

# Wait until any key is pressed, then exit
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the selected points
print("Selected points in left image:")
print(points_left)
print("Selected points in right image:")
print(points_right)
