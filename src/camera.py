# Import required libraries
import math
import numpy as np                                                              # type: ignore
from matrix_operations import *


class Camera:

    def __init__(self, width=1600, height=900, angle=0, speed=0.001, position=np.array([0, 0, -5, 1]), near_plane=0.1, far_plane=100):

        self.angle = angle
        self.speed = speed
        self.width = width
        self.height = height
        self.position = position
        self.h_fov = math.pi / 3
        self.v_fov = self.h_fov * ( self.height / self.width )
        self.far_plane = far_plane
        self.near_plane = near_plane
        self.on_init_matrix()

    def on_init_matrix(self):

        # CAMERA MATRIX
        x, y, z = self.position[:3]
        self.camera_matrix = MatrixOperations.translate(x, y, z)
        
        # self.camera_matrix = (
        #     MatrixOperations.rotate_yaw(math.radians(0))
        #     @ MatrixOperations.rotate_pitch(math.radians(0))
        #     @ MatrixOperations.translate(self.position[:3])
        # )

        # SCREEN MATRIX
        hw, hh = self.width // 2, self.height // 2
        self.screen_matrix = np.array([
            [ hw,   0, 0, 0 ],
            [  0, -hh, 0, 0 ],
            [  0,   0, 1, 0 ],
            [ hw,  hh, 0, 1 ],
        ])

        # PROJECTION MATRIX
        h_fov, v_fov, near, far = self.h_fov, self.v_fov, self.near_plane, self.far_plane
        m00 = 1 / math.tan(h_fov / 2)
        m11 = 1 / math.tan(v_fov / 2)
        m22 = -(far + near) / (far - near)
        m32 = -2 * near * far / (far - near)

        self.projection_matrix = np.array([
            [m00,   0,   0,  0],
            [  0, m11,   0,  0],
            [  0,   0, m22, -1],
            [  0,   0, m32,  0],
        ])

    def get_camera_matrix(self):
        return self. camera_matrix
    
    def get_screen_matrix(self):
        return self.screen_matrix
    
    def get_projection_matrix(self):
        return self.projection_matrix
