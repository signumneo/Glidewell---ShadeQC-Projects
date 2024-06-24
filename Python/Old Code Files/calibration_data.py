import json
import numpy as np


class CalibrationData:
    def __init__(
        self,
        camera_matrix,
        distortion_coefficients,
        rotation_matrix=None,
        translation_vector=None,
        essential_matrix=None,
        fundamental_matrix=None,
    ):
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.essential_matrix = essential_matrix
        self.fundamental_matrix = fundamental_matrix

    @classmethod
    def load(cls, file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                camera_matrix = np.array(data["camera_matrix"])
                distortion_coefficients = np.array(data["distortion_coefficients"])

                # Handle stereo calibration data if present
                rotation_matrix = np.array(data.get("rotation_matrix"))
                translation_vector = np.array(data.get("translation_vector"))
                essential_matrix = np.array(data.get("essential_matrix"))
                fundamental_matrix = np.array(data.get("fundamental_matrix"))

                return cls(
                    camera_matrix,
                    distortion_coefficients,
                    rotation_matrix,
                    translation_vector,
                    essential_matrix,
                    fundamental_matrix,
                )
        except Exception as e:
            print(f"Failed to load calibration data from {file_path}: {e}")
            return None
