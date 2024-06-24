import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load YOLO model for object detection
net = cv2.dnn.readNet(
    r"C:\Users\jacob.thomas\Downloads\yolov3.weights",
    r"C:\Users\jacob.thomas\Downloads\yolov3.cfg",
)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Function to detect objects
def detect_objects(img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []

    for i in range(len(boxes)):
        if i in indexes:
            box = boxes[i]
            detected_objects.append((box[0], box[1], box[2], box[3]))

    return detected_objects


# Load images
left_img_path = r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_left_1.jpg"
right_img_path = r"C:\Users\jacob.thomas\Desktop\Jacob Thomas_118577\Full Time 2024\ShadeQC\IDS camera - GUI\Python\CapturedImages\Pairs\pair_right_1.jpg"
left_img = cv2.imread(left_img_path)
right_img = cv2.imread(right_img_path)

# Detect objects
detected_objects_left = detect_objects(left_img)
detected_objects_right = detect_objects(right_img)

# Draw bounding boxes
for x, y, w, h in detected_objects_left:
    cv2.rectangle(left_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
for x, y, w, h in detected_objects_right:
    cv2.rectangle(right_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show images with bounding boxes
cv2.namedWindow("Left Image with Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Left Image with Detections", left_img)
cv2.resizeWindow("Left Image with Detections", 800, 800)

cv2.namedWindow("Right Image with Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Right Image with Detections", right_img)
cv2.resizeWindow("Right Image with Detections", 800, 800)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Convert to grayscale for stereo processing
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Stereo matching for depth estimation
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left_gray, right_gray)

# Reproject to 3D
Q = np.load("Q.npy")
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Mask disparity values
mask = disparity > disparity.min()
out_points = points_3D[mask]
out_colors = left_img[mask]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    out_points[:, 0], out_points[:, 1], out_points[:, 2], c=out_colors / 255, s=0.1
)
plt.show()

# Save disparity map
cv2.imwrite("disparity_map.jpg", disparity)
