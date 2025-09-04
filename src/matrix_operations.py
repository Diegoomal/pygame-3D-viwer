# Import required libraries
import math
import numpy as np      # type: ignore
from numba import njit  # type: ignore


class MatrixOperations:
    
    @staticmethod
    @njit(fastmath=True)
    def is_out_of_bounds(arr, width, height):
        return np.any(
            (arr[:, 0] < 0)     | 
            (arr[:, 0] > width) | 
            (arr[:, 1] < 0)     | 
            (arr[:, 1] > height)
        )

    @staticmethod
    def translate(position):
        x, y, z = position
        return np.array([
            [ 1, 0, 0, 0 ],
            [ 0, 1, 0, 0 ],
            [ 0, 0, 1, 0 ],
            [ x, y, z, 1 ]
        ])

    @staticmethod
    def scale(n):
        return np.array([
            [ n, 0, 0, 0 ],
            [ 0, n, 0, 0 ],
            [ 0, 0, n, 0 ],
            [ 0, 0, 0, 1 ]
        ])

    @staticmethod
    def rotate_pitch(angle):
        s, i, c = math.sin(angle), -math.sin(angle), math.cos(angle)
        return np.array([
            [ 1, 0, 0, 0 ],
            [ 0, c, s, 0 ],
            [ 0, i, c, 0 ],
            [ 0, 0, 0, 1 ]
        ])

    @staticmethod
    def rotate_yaw(angle):
        s, i, c = math.sin(angle), -math.sin(angle), math.cos(angle)
        return np.array([
            [ c, 0, i, 0 ],
            [ 0, 1, 0, 0 ],
            [ s, 0, c, 0 ],
            [ 0, 0, 0, 1 ]
        ])

    @staticmethod
    def rotate_roll(angle):
        s, i, c = math.sin(angle), -math.sin(angle), math.cos(angle)
        return np.array([
            [ c, s, 0, 0 ],
            [ i, c, 0, 0 ],
            [ 0, 0, 1, 0 ],
            [ 0, 0, 0, 1 ]
        ])
