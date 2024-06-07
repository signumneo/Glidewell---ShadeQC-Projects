import sys
import os
import time
import cv2
import numpy as np
from calibration_processor import CalibrationProcessor
from calibration_data import CalibrationData
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QWidget,
    QPushButton,
)
from PySide6.QtGui import QImage
from PySide6.QtCore import Qt, Slot, QTimer
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
from ids_peak import ids_peak_ipl_extension
from display import Display

VERSION = "1.4.0"
FPS_LIMIT = 30
TARGET_PIXEL_FORMAT = ids_peak_ipl.PixelFormatName_BGRa8


class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        self.widget = QWidget(self)
        self.__main_layout = QVBoxLayout()
        self.widget.setLayout(self.__main_layout)
        self.setCentralWidget(self.widget)

        # Create horizontal layout for the camera feeds
        self.__camera_layout = QHBoxLayout()

        # Create vertical layout for left camera feed with label
        self.__left_layout = QVBoxLayout()
        self.__left_label = QLabel("Left Camera", self)
        self.__left_display = Display()
        self.__left_layout.addWidget(self.__left_label)
        self.__left_layout.addWidget(self.__left_display)

        # Create vertical layout for right camera feed with label
        self.__right_layout = QVBoxLayout()
        self.__right_label = QLabel("Right Camera", self)
        self.__right_display = Display()
        self.__right_layout.addWidget(self.__right_label)
        self.__right_layout.addWidget(self.__right_display)

        # Add left and right layouts to the main camera layout
        self.__camera_layout.addLayout(self.__left_layout)
        self.__camera_layout.addLayout(self.__right_layout)

        # Add camera layout to the main layout
        self.__main_layout.addLayout(self.__camera_layout)

        self.__devices = []
        self.__nodemaps_remote_devices = []
        self.__datastreams = []

        self.__displays = [self.__left_display, self.__right_display]
        self.__acquisition_timer = QTimer()
        self.__frame_counters = [0, 0]
        self.__error_counters = [0, 0]
        self.__acquisition_running = False

        self.__label_infos = None
        self.__label_version = None
        self.__label_aboutqt = None

        self.__image_converters = []

        self.capture_count = 0  # Initialize the capture count

        # Initialize peak library
        ids_peak.Library.Initialize()

        if self.__open_devices():
            try:
                if not self.__start_acquisition():
                    QMessageBox.critical(
                        self, "Error", "Unable to start acquisition!", QMessageBox.Ok
                    )
            except Exception as e:
                QMessageBox.critical(self, "Exception", str(e), QMessageBox.Ok)
        else:
            self.__destroy_all()
            sys.exit(0)

        self.__create_statusbar()
        self.__create_buttons()
        self.setMinimumSize(700, 500)

    def __del__(self):
        self.__destroy_all()

    def __destroy_all(self):
        # Stop acquisition
        self.__stop_acquisition()

        # Close device and peak library
        self.__close_device()
        ids_peak.Library.Close()

    def __open_devices(self):
        try:
            # Create instance of the device manager
            device_manager = ids_peak.DeviceManager.Instance()

            # Update the device manager
            device_manager.Update()

            # Return if no device was found
            if device_manager.Devices().empty():
                QMessageBox.critical(self, "Error", "No device found!", QMessageBox.Ok)
                return False

            # Open the first two openable devices in the manager's device list
            for device in device_manager.Devices():
                if device.IsOpenable():
                    opened_device = device.OpenDevice(ids_peak.DeviceAccessType_Control)
                    self.__devices.append(opened_device)

                    # Open standard data stream
                    datastreams = opened_device.DataStreams()
                    if datastreams.empty():
                        QMessageBox.critical(
                            self,
                            "Error",
                            f"Device {opened_device} has no DataStream!",
                            QMessageBox.Ok,
                        )
                        continue
                    self.__datastreams.append(datastreams[0].OpenDataStream())

                    # Get nodemap of the remote device for all accesses to the GenICam nodemap tree
                    self.__nodemaps_remote_devices.append(
                        opened_device.RemoteDevice().NodeMaps()[0]
                    )

                    # To prepare for untriggered continuous image acquisition, load the default user set if available and wait until execution is finished
                    try:
                        nodemap_remote_device = self.__nodemaps_remote_devices[-1]
                        nodemap_remote_device.FindNode(
                            "UserSetSelector"
                        ).SetCurrentEntry("Default")
                        nodemap_remote_device.FindNode("UserSetLoad").Execute()
                        nodemap_remote_device.FindNode("UserSetLoad").WaitUntilDone()
                    except ids_peak.Exception:
                        # Userset is not available
                        pass

                    # Get the payload size for correct buffer allocation
                    payload_size = (
                        self.__nodemaps_remote_devices[-1]
                        .FindNode("PayloadSize")
                        .Value()
                    )

                    # Get minimum number of buffers that must be announced
                    buffer_count_max = self.__datastreams[
                        -1
                    ].NumBuffersAnnouncedMinRequired()

                    # Allocate and announce image buffers and queue them
                    for i in range(buffer_count_max):
                        buffer = self.__datastreams[-1].AllocAndAnnounceBuffer(
                            payload_size
                        )
                        self.__datastreams[-1].QueueBuffer(buffer)

                if len(self.__devices) == 2:
                    break

            # Return if less than two devices could be opened
            if len(self.__devices) < 2:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Less than two devices could be opened!",
                    QMessageBox.Ok,
                )
                return False

            return True
        except ids_peak.Exception as e:
            QMessageBox.critical(self, "Exception", str(e), QMessageBox.Ok)

        return False

    def __close_device(self):
        """
        Stop acquisition if still running and close datastream and nodemap of the device
        """
        # Stop Acquisition in case it is still running
        self.__stop_acquisition()

        # If a datastream has been opened, try to revoke its image buffers
        if self.__datastream is not None:
            try:
                for buffer in self.__datastream.AnnouncedBuffers():
                    self.__datastream.RevokeBuffer(buffer)
            except Exception as e:
                QMessageBox.information(self, "Exception", str(e), QMessageBox.Ok)

    def __start_acquisition(self):
        """
        Start Acquisition on camera and start the acquisition timer to receive and display images

        :return: True/False if acquisition start was successful
        """
        # Check that devices are opened and that the acquisition is NOT running. If not, return.
        if not self.__devices:
            return False
        if self.__acquisition_running:
            return True

        try:
            for device_index, device in enumerate(self.__devices):
                # Get the maximum framerate possible, limit it to the configured FPS_LIMIT. If the limit can't be reached, set acquisition interval to the maximum possible framerate
                try:
                    max_fps = (
                        self.__nodemaps_remote_devices[device_index]
                        .FindNode("AcquisitionFrameRate")
                        .Maximum()
                    )
                    target_fps = min(max_fps, FPS_LIMIT)
                    self.__nodemaps_remote_devices[device_index].FindNode(
                        "AcquisitionFrameRate"
                    ).SetValue(target_fps)
                except ids_peak.Exception:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "Unable to limit fps, since the AcquisitionFrameRate Node is"
                        " not supported by the connected camera. Program will continue without limit.",
                    )

                # Lock critical features to prevent them from changing during acquisition
                self.__nodemaps_remote_devices[device_index].FindNode(
                    "TLParamsLocked"
                ).SetValue(1)

                image_width = (
                    self.__nodemaps_remote_devices[device_index]
                    .FindNode("Width")
                    .Value()
                )
                image_height = (
                    self.__nodemaps_remote_devices[device_index]
                    .FindNode("Height")
                    .Value()
                )
                input_pixel_format = ids_peak_ipl.PixelFormat(
                    self.__nodemaps_remote_devices[device_index]
                    .FindNode("PixelFormat")
                    .CurrentEntry()
                    .Value()
                )

                # Pre-allocate conversion buffers to speed up first image conversion while the acquisition is running
                image_converter = ids_peak_ipl.ImageConverter()
                image_converter.PreAllocateConversion(
                    input_pixel_format, TARGET_PIXEL_FORMAT, image_width, image_height
                )
                self.__image_converters.append(image_converter)

                # Start acquisition on camera
                self.__datastreams[device_index].StartAcquisition()
                self.__nodemaps_remote_devices[device_index].FindNode(
                    "AcquisitionStart"
                ).Execute()
                self.__nodemaps_remote_devices[device_index].FindNode(
                    "AcquisitionStart"
                ).WaitUntilDone()

            # Setup acquisition timer accordingly
            self.__acquisition_timer.setInterval((1 / target_fps) * 1000)
            self.__acquisition_timer.setSingleShot(False)
            self.__acquisition_timer.timeout.connect(self.on_acquisition_timer)
            self.__acquisition_timer.start()
            self.__acquisition_running = True

            return True

        except Exception as e:
            print("Exception: " + str(e))
            return False

    def __stop_acquisition(self):
        """
        Stop acquisition timer and stop acquisition on camera
        :return:
        """
        # Check that a device is opened and that the acquisition is running. If not, return.
        if not self.__devices or not self.__acquisition_running:
            return

        # Otherwise try to stop acquisition
        try:
            for device_index, device in enumerate(self.__devices):
                self.__nodemaps_remote_devices[device_index].FindNode(
                    "AcquisitionStop"
                ).Execute()

                # Stop and flush datastream
                self.__datastreams[device_index].KillWait()
                self.__datastreams[device_index].StopAcquisition(
                    ids_peak.AcquisitionStopMode_Default
                )
                self.__datastreams[device_index].Flush(
                    ids_peak.DataStreamFlushMode_DiscardAll
                )

                # Unlock parameters after acquisition stop
                self.__nodemaps_remote_devices[device_index].FindNode(
                    "TLParamsLocked"
                ).SetValue(0)

            self.__acquisition_running = False

        except Exception as e:
            QMessageBox.information(self, "Exception", str(e), QMessageBox.Ok)

    def __create_statusbar(self):
        status_bar = QWidget(self.centralWidget())
        status_bar_layout = QHBoxLayout()
        status_bar_layout.setContentsMargins(0, 0, 0, 0)

        self.__label_infos = QLabel(status_bar)
        self.__label_infos.setAlignment(Qt.AlignLeft)
        status_bar_layout.addWidget(self.__label_infos)
        status_bar_layout.addStretch()

        self.__label_version = QLabel(status_bar)
        self.__label_version.setText("simple_live_qtwidgets v" + VERSION)
        self.__label_version.setAlignment(Qt.AlignRight)
        status_bar_layout.addWidget(self.__label_version)

        self.__label_aboutqt = QLabel(status_bar)
        self.__label_aboutqt.setObjectName("aboutQt")
        self.__label_aboutqt.setText("<a href='#aboutQt'>About Qt</a>")
        self.__label_aboutqt.setAlignment(Qt.AlignRight)
        self.__label_aboutqt.linkActivated.connect(self.on_aboutqt_link_activated)
        status_bar_layout.addWidget(self.__label_aboutqt)
        status_bar.setLayout(status_bar_layout)

        self.__main_layout.addWidget(status_bar)

    # Adding buttons for all major functionalities

    def __create_buttons(self):
        self.capture_button = QPushButton("Capture Image Pair", self)
        self.capture_button.clicked.connect(self.capture_image)
        self.__main_layout.addWidget(self.capture_button)

        self.capture_left_button = QPushButton("Capture Left Image", self)
        self.capture_left_button.clicked.connect(self.capture_left_image)
        self.__main_layout.addWidget(self.capture_left_button)

        self.capture_right_button = QPushButton("Capture Right Image", self)
        self.capture_right_button.clicked.connect(self.capture_right_image)
        self.__main_layout.addWidget(self.capture_right_button)

        self.calibrate_individual_button = QPushButton(
            "Calibrate Individual Cameras", self
        )
        self.calibrate_individual_button.clicked.connect(
            self.calibrate_individual_cameras
        )
        self.__main_layout.addWidget(self.calibrate_individual_button)

        self.calibrate_stereo_button = QPushButton("Calibrate Stereo", self)
        self.calibrate_stereo_button.clicked.connect(self.calibrate_stereo)
        self.__main_layout.addWidget(self.calibrate_stereo_button)

        self.rectify_stereo_button = QPushButton("Rectify Stereo", self)
        self.rectify_stereo_button.clicked.connect(self.rectify_stereo)
        self.__main_layout.addWidget(self.rectify_stereo_button)

        self.visualize_disparity_button = QPushButton("Visualize Disparity", self)
        self.visualize_disparity_button.clicked.connect(self.visualize_disparity)
        self.__main_layout.addWidget(self.visualize_disparity_button)

        self.visualize_3d_button = QPushButton("Visualize 3D Points", self)
        self.visualize_3d_button.clicked.connect(self.visualize_3d_points)
        self.__main_layout.addWidget(self.visualize_3d_button)

    def capture_image(self):
        images = self.acquire_images()
        if images is not None:
            output_dir = "CapturedImages"
            left_dir = os.path.join(output_dir, "LeftCamera")
            right_dir = os.path.join(output_dir, "RightCamera")
            pair_dir = os.path.join(output_dir, "Pairs")

            # Ensure the directories exist
            os.makedirs(left_dir, exist_ok=True)
            os.makedirs(right_dir, exist_ok=True)
            os.makedirs(pair_dir, exist_ok=True)

            self.capture_count += 1  # Increment the capture count

            for i, image in enumerate(images):
                if i == 0:
                    # Save left camera image
                    file_path = os.path.join(
                        left_dir, f"left_image_{self.capture_count}.jpg"
                    )
                    cv2.imwrite(file_path, image)
                elif i == 1:
                    # Save right camera image
                    file_path = os.path.join(
                        right_dir, f"right_image_{self.capture_count}.jpg"
                    )
                    cv2.imwrite(file_path, image)

            # Save the pair of images
            pair_path_left = os.path.join(
                pair_dir, f"pair_left_{self.capture_count}.jpg"
            )
            pair_path_right = os.path.join(
                pair_dir, f"pair_right_{self.capture_count}.jpg"
            )
            cv2.imwrite(pair_path_left, images[0])
            cv2.imwrite(pair_path_right, images[1])

            QMessageBox.information(
                self, "Image Captured", f"Images saved to {output_dir}"
            )

    def capture_left_image(self):
        image = self.acquire_images(single_camera=0)
        if image is not None:
            output_dir = "CapturedImages"
            left_dir = os.path.join(output_dir, "LeftCamera")

            # Ensure the directories exist
            os.makedirs(left_dir, exist_ok=True)

            self.capture_count += 1  # Increment the capture count

            file_path = os.path.join(left_dir, f"left_image_{self.capture_count}.jpg")
            cv2.imwrite(file_path, image)
            QMessageBox.information(
                self, "Image Captured", f"Image saved to {file_path}"
            )

    def capture_right_image(self):
        image = self.acquire_images(single_camera=1)
        if image is not None:
            output_dir = "CapturedImages"
            right_dir = os.path.join(output_dir, "RightCamera")

            # Ensure the directories exist
            os.makedirs(right_dir, exist_ok=True)

            self.capture_count += 1  # Increment the capture count

            file_path = os.path.join(right_dir, f"right_image_{self.capture_count}.jpg")
            cv2.imwrite(file_path, image)
            QMessageBox.information(
                self, "Image Captured", f"Image saved to {file_path}"
            )

    def calibrate_individual_cameras(self):
        calibration_processor = CalibrationProcessor()
        calibration_processor.perform_individual_calibration()
        self.display_calibration_metrics("LeftCamera")
        self.display_calibration_metrics("RightCamera")

    def calibrate_stereo(self):
        calibration_processor = CalibrationProcessor()
        calibration_processor.perform_stereo_calibration()
        self.display_calibration_metrics("Stereo")

    def rectify_stereo(self):
        calibration_processor = CalibrationProcessor()
        left_rectified, right_rectified, Q = calibration_processor.stereo_rectify()
        self.left_rectified = left_rectified
        self.right_rectified = right_rectified
        self.Q = Q
        QMessageBox.information(
            self, "Stereo Rectification", "Stereo rectification completed."
        )

    def visualize_disparity(self):
        if hasattr(self, "left_rectified") and hasattr(self, "right_rectified"):
            calibration_processor = CalibrationProcessor()
            disparity = calibration_processor.visualize_disparity(
                self.left_rectified, self.right_rectified
            )
            self.disparity = disparity
            QMessageBox.information(
                self, "Disparity Visualization", "Disparity visualization completed."
            )
        else:
            QMessageBox.warning(self, "Error", "Please rectify stereo images first.")

    def visualize_3d_points(self):
        if hasattr(self, "disparity") and hasattr(self, "Q"):
            calibration_processor = CalibrationProcessor()
            output_points, output_colors = calibration_processor.visualize_3d_points(
                self.disparity, self.Q
            )
            QMessageBox.information(
                self, "3D Points Visualization", "3D points visualization completed."
            )
        else:
            QMessageBox.warning(self, "Error", "Please visualize disparity first.")

    def display_calibration_metrics(self, camera):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(
            base_dir,
            (
                f"calibration_data_{camera.lower()}.json"
                if camera != "Stereo"
                else "calibration_data_stereo.json"
            ),
        )

        print(f"Trying to load calibration data from {file_path}")  # Debug print

        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")  # Debug print
            QMessageBox.information(
                self, "Calibration Metrics", f"No calibration data found for {camera}."
            )
            return

        calibration_data = CalibrationData.load(file_path)
        if calibration_data is not None:
            if camera == "Stereo":
                # Display stereo calibration data
                metrics = (
                    f"Rotation Matrix:\n{calibration_data.rotation_matrix}\n\n"
                    f"Translation Vector:\n{calibration_data.translation_vector}\n\n"
                    f"Essential Matrix:\n{calibration_data.essential_matrix}\n\n"
                    f"Fundamental Matrix:\n{calibration_data.fundamental_matrix}"
                )
            else:
                # Format the calibration data to 2 decimal places
                camera_matrix = np.array(calibration_data.camera_matrix)
                distortion_coefficients = np.array(
                    calibration_data.distortion_coefficients
                )

                formatted_camera_matrix = [
                    [round(float(val), 2) for val in row] for row in camera_matrix
                ]
                formatted_distortion_coefficients = [
                    round(float(val), 2) for val in distortion_coefficients[0]
                ]

                # Display the calibration data using QMessageBox or another appropriate method
                metrics = (
                    f"Camera Matrix ({camera}):\n{formatted_camera_matrix}\n\n"
                    f"Distortion Coefficients ({camera}):\n{formatted_distortion_coefficients}\n\n"
                )
            QMessageBox.information(self, "Calibration Metrics", metrics)
        else:
            print(f"Failed to load calibration data from {file_path}")  # Debug print
            QMessageBox.information(
                self, "Calibration Metrics", "No calibration data found."
            )

    def acquire_images(self, single_camera=None):
        """
        Acquire a single image from each camera and return them as a list of numpy arrays.
        """
        images = []
        if single_camera is not None:
            device_index = single_camera
            try:
                # Get buffer from device's datastream
                buffer = self.__datastreams[device_index].WaitForFinishedBuffer(5000)

                # Create IDS peak IPL image for debayering and convert it to RGBa8 format
                ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
                converted_ipl_image = self.__image_converters[device_index].Convert(
                    ipl_image, TARGET_PIXEL_FORMAT
                )

                # Queue buffer so that it can be used again
                self.__datastreams[device_index].QueueBuffer(buffer)

                # Get raw image data from converted image and construct a numpy array from it
                image_np_array = converted_ipl_image.get_numpy_1D()
                image = np.array(image_np_array).reshape(
                    (converted_ipl_image.Height(), converted_ipl_image.Width(), 4)
                )

                # Rotate the image 90 degrees clockwise
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

                return image
            except ids_peak.Exception as e:
                QMessageBox.critical(self, "Exception", str(e), QMessageBox.Ok)
                return None
        else:
            for device_index in range(2):
                try:
                    # Get buffer from device's datastream
                    buffer = self.__datastreams[device_index].WaitForFinishedBuffer(
                        5000
                    )

                    # Create IDS peak IPL image for debayering and convert it to RGBa8 format
                    ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
                    converted_ipl_image = self.__image_converters[device_index].Convert(
                        ipl_image, TARGET_PIXEL_FORMAT
                    )

                    # Queue buffer so that it can be used again
                    self.__datastreams[device_index].QueueBuffer(buffer)

                    # Get raw image data from converted image and construct a numpy array from it
                    image_np_array = converted_ipl_image.get_numpy_1D()
                    image = np.array(image_np_array).reshape(
                        (converted_ipl_image.Height(), converted_ipl_image.Width(), 4)
                    )

                    # Rotate the image 90 degrees clockwise
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

                    images.append(image)
                except ids_peak.Exception as e:
                    QMessageBox.critical(self, "Exception", str(e), QMessageBox.Ok)
                    images.append(None)
            return images

    def update_counters(self):
        """
        This function gets called when the frame and error counters have changed
        :return:
        """
        self.__label_infos.setText(
            "Acquired: "
            + str(self.__frame_counters[0])
            + ", Errors: "
            + str(self.__error_counters[0])
            + " | "
            + "Acquired: "
            + str(self.__frame_counters[1])
            + ", Errors: "
            + str(self.__error_counters[1])
        )

    @Slot()
    def on_acquisition_timer(self):
        """
        This function gets called on every timeout of the acquisition timer
        """
        try:
            for device_index in range(2):
                # Get buffer from device's datastream
                buffer = self.__datastreams[device_index].WaitForFinishedBuffer(5000)

                # Create IDS peak IPL image for debayering and convert it to RGBa8 format
                ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
                # NOTE: Use `ImageConverter`, since the `ConvertTo` function re-allocates
                #       the conversion buffers on every call
                converted_ipl_image = self.__image_converters[device_index].Convert(
                    ipl_image, TARGET_PIXEL_FORMAT
                )

                # Queue buffer so that it can be used again
                self.__datastreams[device_index].QueueBuffer(buffer)

                # Get raw image data from converted image and construct a QImage from it
                image_np_array = converted_ipl_image.get_numpy_1D()
                image = np.array(image_np_array).reshape(
                    (converted_ipl_image.Height(), converted_ipl_image.Width(), 4)
                )

                # Rotate the image 90 degrees clockwise
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

                qimage = QImage(
                    image.data, image.shape[1], image.shape[0], QImage.Format_RGB32
                )

                # Make an extra copy of the QImage to make sure that memory is copied and can't get overwritten later on
                qimage_cpy = qimage.copy()

                # Emit signal that the image is ready to be displayed
                self.__displays[device_index].on_image_received(qimage_cpy)
                self.__displays[device_index].update()

                # Increase frame counter
                self.__frame_counters[device_index] += 1
        except ids_peak.Exception as e:
            self.__error_counters[device_index] += 1
            print("Exception: " + str(e))

        # Update counters
        self.update_counters()

    @Slot(str)
    def on_aboutqt_link_activated(self, link):
        if link == "#aboutQt":
            QMessageBox.aboutQt(self, "About Qt")
