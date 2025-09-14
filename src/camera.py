import math
import numpy as np                                                              # type: ignore
from matrix_operations import MatrixOperations


class Camera:

    def __init__(self, 
                 width, height, 
                 position=np.array([0,0,-5,1]), 
                 h_fov=math.pi/3, near=0.1, far=100
                ):
        self.width, self.height = width, height
        self.position = position
        self.h_fov, self.v_fov = h_fov, h_fov*(height/width)
        self.near, self.far = near, far

    def get_view_matrix(self):
        x, y, z = self.position[:3]
        return MatrixOperations.translate(x, y, z)

    def get_projection_matrix(self):

        m00 = 1/math.tan(self.h_fov/2)
        m11 = 1/math.tan(self.v_fov/2)
        m22 = -(self.far+self.near)/(self.far-self.near)
        m32 = -2*self.near*self.far/(self.far-self.near)
        
        return np.array([
            [ m00,   0,   0,  0 ],
            [   0, m11,   0,  0 ],
            [   0,   0, m22, -1 ],
            [   0,   0, m32,  0 ]
        ])

    def get_screen_matrix(self):

        hw, hh = self.width//2, self.height//2
        
        return np.array([
            [ hw,   0, 0, 0 ],
            [  0, -hh, 0, 0 ],
            [  0,   0, 1, 0 ],
            [ hw,  hh, 0, 1 ]
        ])
