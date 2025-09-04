# Import required libraries
import numpy as np      # type: ignore
import pygame as pg     # type: ignore
from camera import *
from matrix_operations import *

class Polygon:

    def __init__(self, faces:list[list[int]]=[], vertices:list[list[float]]=[], speed:float=0.01):
        self.faces:     list[list[int  ]]   = faces
        self.vertices:  list[list[float]]   = vertices
        self.speed:     float               = speed

    def process(self, camera: Camera) -> 'Polygon':

        if camera is None: raise ValueError("Camera cannot be None")

        # self.vertices = self.vertices @ MatrixOperations.translate([0.0001, 0.0001, 0.0001])

        transformed_vertices = (
            np.asarray(self.vertices)
            @ camera.get_camera_matrix()
            @ camera.get_projection_matrix()
        ) 
        
        self.transformed_vertices_3d = transformed_vertices.copy()

        transformed_vertices /= transformed_vertices[:, -1].reshape(-1, 1)      # Normalize by the homogeneous coordinate
        transformed_vertices = transformed_vertices @ camera.get_screen_matrix() # Apply screen transformation
        transformed_vertices = transformed_vertices[:, :2]                      # Keep only 2D coordinates

        self.transformed_vertices_2d = transformed_vertices.copy()

        return self

    def get_transformed_vertices_2d(self):
        if self.transformed_vertices_2d is None: 
            raise ValueError("transformed_vertices_2d cannot be None")
        return self.transformed_vertices_2d

    def get_transformed_vertices_3d(self):
        if self.transformed_vertices_3d is None: 
            raise ValueError("transformed_vertices_3d cannot be None")
        return self.transformed_vertices_3d

    def auto_rotation(self):
        x, y, z = [0.0001, 0.0001, 0.0001]
        self.vertices = self.vertices @ MatrixOperations.translate(x, y, z)
        self.vertices = self.vertices @ MatrixOperations.rotate_yaw(self.speed)
        self.vertices = self.vertices @ MatrixOperations.rotate_roll(self.speed)
        self.vertices = self.vertices @ MatrixOperations.rotate_pitch(self.speed)

    def user_input_handler(self):
        key = pg.key.get_pressed()

        # X-Axis
        if key[pg.K_a]: self.vertices -= self.speed * np.array([1, 0, 0, 0])    # Left
        if key[pg.K_d]: self.vertices += self.speed * np.array([1, 0, 0, 0])    # Right

        # Y-Axis
        if key[pg.K_w]: self.vertices -= self.speed * np.array([0, 1, 0, 0])    # forward
        if key[pg.K_s]: self.vertices += self.speed * np.array([0, 1, 0, 0])    # back

        # Z-Axis
        if key[pg.K_e]: self.vertices += (self.speed * 10) * np.array([0, 0, 1, 0]) # Zoom In
        if key[pg.K_q]: self.vertices -= (self.speed * 10) * np.array([0, 0, 1, 0]) # Zoom Out

    def update(self):
        self.auto_rotation()
        # self.user_input_handler()   
