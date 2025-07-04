# Import required libraries
import math
import numpy as np      # type: ignore
import pygame as pg     # type: ignore
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
        self.camera_matrix = MatrixOperations.translate(self.position[:3])
        
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

class Polygon:

    def __init__(self, faces, vertices, uvs=None, uv_faces=None, speed=0.01):
        self.faces = faces
        self.vertices = vertices
        self.uvs = uvs or []
        self.uv_faces = uv_faces or []
        self.speed = speed

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
        self.vertices = self.vertices @ MatrixOperations.translate([0.0001, 0.0001, 0.0001])
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

class FileManager:

    def __init__(self, filepath):
        self.filepath = filepath
        self.faces: list[list[int]] = []
        self.vertices: list[list[float]] = []
        self.uvs: list[list[float]] = []
        self.uv_faces: list[list[int]] = []

    def load(self) -> 'FileManager':
        with open(self.filepath) as f:
            for line in f:
                if line.startswith('v '):
                    self.vertices.append([float(i) for i in line.split()[1:]] + [1])
                elif line.startswith('vt '):
                    self.uvs.append([float(i) for i in line.split()[1:]])
                elif line.startswith('f '):
                    v_idx = []
                    vt_idx = []
                    for token in line.strip().split()[1:]:
                        parts = token.split('/')
                        v_idx.append(int(parts[0]) - 1)
                        vt_idx.append(int(parts[1]) - 1)
                    self.faces.append(v_idx)
                    self.uv_faces.append(vt_idx)
        return self

    def get_faces(self) -> list[list[int]]:
        return self.faces
    
    def get_vertices(self) -> list[list[float]]:
        return self.vertices

    def get_dict(self) -> dict:
        return { 'faces': self.faces, 'vertices': self.vertices }
    
    def get_polygon(self) -> Polygon:
        return Polygon(self.faces, self.vertices, self.uvs, self.uv_faces)

class Render:

    def __init__(self, width=1600, height=900):
        self.width = width
        self.height = height

    def textured_triangle(self, screen, pts, tex_coords, texture):
        # Bounding box
        xmin = max(min(p[0] for p in pts), 0)
        xmax = min(max(p[0] for p in pts), self.width - 1)
        ymin = max(min(p[1] for p in pts), 0)
        ymax = min(max(p[1] for p in pts), self.height - 1)

        def edge_func(a, b, c):
            return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

        area = edge_func(pts[0], pts[1], pts[2])
        if area == 0:
            return

        texture_width, texture_height = texture.get_size()
        for y in range(int(ymin), int(ymax) + 1):
            for x in range(int(xmin), int(xmax) + 1):
                p = (x + 0.5, y + 0.5)
                w0 = edge_func(pts[1], pts[2], p)
                w1 = edge_func(pts[2], pts[0], p)
                w2 = edge_func(pts[0], pts[1], p)

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    w0 /= area
                    w1 /= area
                    w2 /= area

                    u = w0 * tex_coords[0][0] + w1 * tex_coords[1][0] + w2 * tex_coords[2][0]
                    v = w0 * tex_coords[0][1] + w1 * tex_coords[1][1] + w2 * tex_coords[2][1]
                    tex_x = int(u * texture_width)
                    tex_y = int((1 - v) * texture_height)
                    color = texture.get_at((tex_x, tex_y))
                    screen.set_at((x, y), color)

    def polygon_to_screen(self, screen, polygon: Polygon, texture: pg.Surface = None):
        if polygon is None: raise ValueError("Polygon cannot be None")

        faces = polygon.faces
        uvs = polygon.uvs
        uv_faces = polygon.uv_faces
        tvs2d = polygon.transformed_vertices_2d

        for i, face in enumerate(faces):
            pts = np.array([tvs2d[idx] for idx in face])
            if not MatrixOperations.is_out_of_bounds(pts, self.width, self.height):
                if texture and polygon.uvs and polygon.uv_faces:
                    tex_coords = [uvs[uv_faces[i][j]] for j in range(len(face))]
                    self.textured_triangle(screen, pts, tex_coords, texture)
                else:
                    pg.draw.polygon(screen, pg.Color('orange'), pts, 1)
