import json
import os
import numpy as np


class CalibrationData:
    def __init__(
        self,
        camera_matrix=None,
        distortion_coefficients=None,
        rotation_vectors=None,
        translation_vectors=None,
        rotation_matrix=None,
        translation_vector=None,
        essential_matrix=None,
        fundamental_matrix=None,
    ):
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.rotation_vectors = rotation_vectors
        self.translation_vectors = translation_vectors
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.essential_matrix = essential_matrix
        self.fundamental_matrix = fundamental_matrix

    def save(self, file_path):
        data = {
            "camera_matrix": (
                self.camera_matrix.tolist() if self.camera_matrix is not None else None
            ),
            "distortion_coefficients": (
                self.distortion_coefficients.tolist()
                if self.distortion_coefficients is not None
                else None
            ),
            "rotation_vectors": (
                [vec.tolist() for vec in self.rotation_vectors]
                if self.rotation_vectors is not None
                else None
            ),
            "translation_vectors": (
                [vec.tolist() for vec in self.translation_vectors]
                if self.translation_vectors is not None
                else None
            ),
            "rotation_matrix": (
                self.rotation_matrix.tolist()
                if self.rotation_matrix is not None
                else None
            ),
            "translation_vector": (
                self.translation_vector.tolist()
                if self.translation_vector is not None
                else None
            ),
            "essential_matrix": (
                self.essential_matrix.tolist()
                if self.essential_matrix is not None
                else None
            ),
            "fundamental_matrix": (
                self.fundamental_matrix.tolist()
                if self.fundamental_matrix is not None
                else None
            ),
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load(file_path):
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            data = json.load(f)
            camera_matrix = (
                np.array(data["camera_matrix"])
                if data.get("camera_matrix") is not None
                else None
            )
            distortion_coefficients = (
                np.array(data["distortion_coefficients"])
                if data.get("distortion_coefficients") is not None
                else None
            )
            rotation_vectors = (
                [np.array(vec) for vec in data["rotation_vectors"]]
                if data.get("rotation_vectors") is not None
                else None
            )
            translation_vectors = (
                [np.array(vec) for vec in data["translation_vectors"]]
                if data.get("translation_vectors") is not None
                else None
            )
            rotation_matrix = (
                np.array(data["rotation_matrix"])
                if data.get("rotation_matrix") is not None
                else None
            )
            translation_vector = (
                np.array(data["translation_vector"])
                if data.get("translation_vector") is not None
                else None
            )
            essential_matrix = (
                np.array(data["essential_matrix"])
                if data.get("essential_matrix") is not None
                else None
            )
            fundamental_matrix = (
                np.array(data["fundamental_matrix"])
                if data.get("fundamental_matrix") is not None
                else None
            )

            return CalibrationData(
                camera_matrix,
                distortion_coefficients,
                rotation_vectors,
                translation_vectors,
                rotation_matrix,
                translation_vector,
                essential_matrix,
                fundamental_matrix,
            )
