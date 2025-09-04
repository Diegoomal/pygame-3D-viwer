# Import required libraries
import math
import numpy as np      # type: ignore
from numba import njit  # type: ignore
import pygame as pg     # type: ignore
import pygame.gfxdraw as gfx # type: ignore


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
        self.faces:    list[list[int]]   = []
        self.vertices: list[list[float]] = []

    def load(self) -> 'FileManager':
        with open(self.filepath) as f:
            for line in f:
                if line.startswith('v '):  # Parse vertices
                    self.vertices.append([float(i) for i in line.split()[1:]] + [1])
                elif line.startswith('f'):  # Parse faces
                    self.faces.append([int(f.split('/')[0]) - 1 for f in line.split()[1:]])
        return self

    def get_faces(self) -> list[list[int]]:
        return self.faces
    
    def get_vertices(self) -> list[list[float]]:
        return self.vertices

    def get_dict(self) -> dict:
        return { 'faces': self.faces, 'vertices': self.vertices }
    
    def get_polygon(self) -> Polygon:
        return Polygon(self.faces, self.vertices)

class Render:

    def __init__(self, width:int=1600, height:int=900, texture:pg.Surface|None=None):
        self.width = width
        self.height = height
        self.texture = texture

    def polygon_to_screen(self, screen, polygon:Polygon):
        if polygon is None:
            raise ValueError("Polygon cannot be None")

        faces, tvs2d = polygon.faces, polygon.transformed_vertices_2d

        for face in faces:
            pts = np.array([tvs2d[i] for i in face])
            if not MatrixOperations.is_out_of_bounds(pts, self.width, self.height):
                if self.texture is None:
                    # wireframe
                    pg.draw.polygon(screen, pg.Color('orange'), pts, 1)
                else:
                    # textured (sem UV → só mapeia textura inteira no polígono)
                    gfx.textured_polygon(screen, pts.astype(int), self.texture, 0, 0)

class App:

    def __init__(self, polygon:Polygon, fps:int=30, width:int=1600, height:int=900):

        pg.init()
        self.fps = fps
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((width, height))

        self.polygon: Polygon = polygon
        self.camera: Camera = Camera(width=width, height=height, position=np.array([0, 0, -5, 1]))

        # self.render: Render = Render(width=width, height=height)

        texture = pg.image.load('./assets/textures/white.png').convert()
        self.render: Render = Render(width=width, height=height, texture=texture)

    def update(self):

        # INPUT HANDLER
        for event in pg.event.get(): 
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE): pg.quit(); exit()

        camera: Camera = Camera(position=np.array([0, 0, -5, 1]))

        self.polygon.update()
        self.polygon.process(camera)
    
    def draw(self):
        self.screen.fill( pg.Color('darkslategray') )
        self.render.polygon_to_screen(self.screen, self.polygon)
        # Update the display and maintain frame rate
        pg.display.set_caption(f"FPS: {self.clock.get_fps():.2f}")
        pg.display.flip()
        self.clock.tick(self.fps)

    def run(self):
        while True:
            self.update()
            self.draw()

# Entry point
if __name__ == '__main__':
    polygon: Polygon = FileManager('./assets/models/suzanne/model.obj').load().get_polygon()
    App(polygon=polygon).run()