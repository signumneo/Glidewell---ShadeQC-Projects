import cv2

# Load the image
image = cv2.imread(
    r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\CapturedImages\image_1.jpg"
)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

if ret:
    print("Checkerboard detected.")
else:
    print("Checkerboard not detected.")
