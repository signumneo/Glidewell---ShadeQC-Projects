import cv2
import os
import time

# Specify the directory where captured frames will be saved
output_folder = "Captured Frames/Calibration"
left_folder = os.path.join(output_folder, "Left")
right_folder = os.path.join(output_folder, "Right")
combined_folder = os.path.join(output_folder, "Combined")

# Create directories if they don't exist
os.makedirs(left_folder, exist_ok=True)
os.makedirs(right_folder, exist_ok=True)
os.makedirs(combined_folder, exist_ok=True)

# Specify the backend API
backend_api = cv2.CAP_DSHOW  # or cv2.CAP_MSMF for Media Foundation

# Initialize both cameras with the specified backend
camera1 = cv2.VideoCapture(0, backend_api)
camera2 = cv2.VideoCapture(1, backend_api)


# Function to set autofocus (simulated with camera properties)
def set_focus(camera, focus_value=0):
    if camera:
        camera.set(
            cv2.CAP_PROP_AUTOFOCUS, 1
        )  # Turn on autofocus (might not work with all cameras)
        camera.set(cv2.CAP_PROP_FOCUS, focus_value)  # Adjust focus (0 to 255)


# Set autofocus for both cameras
set_focus(camera1)
set_focus(camera2)

# Check if both cameras are opened successfully
if not camera1.isOpened():
    print("Camera 1 could not be opened.")
    camera1 = None

if not camera2.isOpened():
    print("Camera 2 could not be opened.")
    camera2 = None

if camera1 is None and camera2 is None:
    print("Neither Camera 1 nor Camera 2 could be opened. Exiting.")
    exit()

capture_interval = 5  # Interval in seconds
last_capture_time = time.time()

frame_count = 0  # Simple naming with frame count

while True:
    # Initialize frames
    frame1 = frame2 = None

    # Capture frame-by-frame from both cameras
    if camera1:
        ret1, frame1 = camera1.read()
        if not ret1:
            print("Failed to grab frame from Camera 1")
            break

    if camera2:
        ret2, frame2 = camera2.read()
        if not ret2:
            print("Failed to grab frame from Camera 2")
            break

    # Resize the frames if both are available
    if frame1 is not None:
        frame1 = cv2.resize(frame1, (640, 480))
    if frame2 is not None:
        frame2 = cv2.resize(frame2, (640, 480))

    # Concatenate frames side by side if both are available
    if frame1 is not None and frame2 is not None:
        concatenated_frame = cv2.hconcat([frame1, frame2])
    elif frame1 is not None:
        concatenated_frame = frame1
    elif frame2 is not None:
        concatenated_frame = frame2
    else:
        break  # If no frames are available, exit the loop

    # Display the concatenated frame
    cv2.imshow("Concatenated Camera Feeds", concatenated_frame)

    # Check if it's time to save the frame
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        frame_count += 1  # Increment frame count for naming

        # Save individual frames
        if frame1 is not None:
            output_path_left = os.path.join(left_folder, f"left_{frame_count}.jpg")
            cv2.imwrite(output_path_left, frame1)
            print(f"Saved left frame at {output_path_left}")

        if frame2 is not None:
            output_path_right = os.path.join(right_folder, f"right_{frame_count}.jpg")
            cv2.imwrite(output_path_right, frame2)
            print(f"Saved right frame at {output_path_right}")

        # Save concatenated frame in the combined folder
        output_path_combined = os.path.join(
            combined_folder, f"combined_{frame_count}.jpg"
        )
        cv2.imwrite(output_path_combined, concatenated_frame)
        print(f"Saved combined frame at {output_path_combined}")

        last_capture_time = current_time

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the cameras and close the window
if camera1:
    camera1.release()
if camera2:
    camera2.release()
cv2.destroyAllWindows()
