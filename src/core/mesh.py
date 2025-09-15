import numpy as np                                                              # type: ignore
from core.matrix_operations import MatrixOperations


class Mesh:

    def __init__(self, faces:np.ndarray, vertices:np.ndarray, position=[0,0,0,1], texture=None):
        self.faces = np.array(faces, dtype=object)                              # int -> object
        self.vertices = np.array(vertices, dtype=float)
        self.model_matrix = MatrixOperations.translate(*position[:3])
        self.texture = texture

    def apply_transform(self, matrix:np.ndarray):
        self.model_matrix = self.model_matrix @ matrix

    def get_transformed(self) -> np.ndarray:
        return self.vertices @ self.model_matrix

    def auto_rotate(self, angle):
        rot_x = MatrixOperations.rotate_pitch(angle)
        rot_y = MatrixOperations.rotate_yaw(angle)
        rot_z = MatrixOperations.rotate_roll(angle)
        self.apply_transform(rot_y @ rot_x @ rot_z)
