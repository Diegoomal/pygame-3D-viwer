import numpy as np                                                              # type: ignore
from numba import njit                                                          # type: ignore


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
    @njit(fastmath=True)
    def translate(x, y, z):
        """
        Gera uma matriz de translação 4x4 que desloca pontos/objetos 3D pelas coordenadas (x, y, z).
        """
        return np.array([
            [ 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 1.0, 0.0 ],
            [   x,   y,   z, 1.0 ]
        ], dtype=np.float64)

    @staticmethod
    @njit(fastmath=True)
    def scale(n):
        return np.array([
            [   n, 0.0, 0.0, 0.0 ],
            [ 0.0,   n, 0.0, 0.0 ],
            [ 0.0, 0.0,   n, 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0 ]
        ], dtype=np.float64)
   
    @staticmethod
    @njit(fastmath=True)
    def rotate_pitch(angle):                                                    # X-axis
        s, c = np.sin(angle), np.cos(angle)
        return np.array([
            [ 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0,   c,   s, 0.0 ],
            [ 0.0,  -s,   c, 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0 ]
        ], dtype=np.float64)

    @staticmethod
    @njit(fastmath=True)
    def rotate_yaw(angle):                                                      # Y-Axis
        s, c = np.sin(angle), np.cos(angle)
        return np.array([
            [   c, 0.0,   -s, 0.0 ],
            [ 0.0, 1.0,  0.0, 0.0 ],
            [   s, 0.0,    c, 0.0 ],
            [ 0.0, 0.0,  0.0, 1.0 ]
        ], dtype=np.float64)
    
    @staticmethod
    @njit(fastmath=True)
    def rotate_roll(angle):                                                     # Z-Axis
        s, c = np.sin(angle), np.cos(angle)
        return np.array([
            [   c,   s, 0.0, 0.0 ],
            [  -s,   c, 0.0, 0.0 ],
            [ 0.0, 0.0, 1.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0 ]
        ], dtype=np.float64)
