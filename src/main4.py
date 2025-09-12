import math
import numpy as np                      # type: ignore
import pygame as pg                     # type: ignore
import pygame.gfxdraw as gfx            # type: ignore
from core.matrix_operations import MatrixOperations


class Mesh:

    def __init__(self, faces:list[list[int]], vertices:list[list[float]], position=[0,0,0,1]):
        self.faces = np.array(faces, dtype=object)                              # int -> object
        self.vertices = np.array(vertices, dtype=float)
        self.model_matrix = MatrixOperations.translate(*position[:3])

    def apply_transform(self, matrix:np.ndarray):
        self.model_matrix = self.model_matrix @ matrix

    def get_transformed(self) -> np.ndarray:
        return self.vertices @ self.model_matrix
    
class Scene:

    def __init__(self):
        self.meshes = []

    def add(self, mesh):
        self.meshes.append(mesh)

    def __iter__(self):
        return iter(self.meshes)
    
class FileManager:

    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):

        faces, vertices = [], []

        with open(self.filepath) as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(i) for i in line.split()[1:]]+[1])
                elif line.startswith('f'):
                    faces.append([int(p.split('/')[0])-1 for p in line.split()[1:]])

        return faces, vertices

class Camera:

    def __init__(self, width, height, position=np.array([0,0,-5,1]), h_fov=math.pi/3, near=0.1, far=100):
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

class LambertShader:

    def __init__(self, light_dir=np.array([0, 0, -1])):
        self.light_dir = light_dir / np.linalg.norm(light_dir)

    def shade(self, verts3d, face):
        pts3d = np.array([verts3d[i][:3] for i in face])
        v1, v2 = pts3d[1] - pts3d[0], pts3d[2] - pts3d[0]
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        intensity = max(0, np.dot(normal, self.light_dir))
        val = int(255 * intensity)
        return (val, val, 0)

class Renderer:

    def __init__(self, screen, width, height, texture=None, shader=None):
        self.screen = screen
        self.width = width
        self.height = height
        self.texture = texture
        self.shader = shader

    def render(self, mesh, camera, render_type='wireframe'):

        verts3d = mesh.get_transformed().copy()

        verts = mesh.get_transformed().copy() @ camera.get_view_matrix() @ camera.get_projection_matrix()
        verts /= verts[:, -1].reshape(-1, 1)
        verts = verts @ camera.get_screen_matrix()
        verts2d = verts[:, :2]

        visible_faces = []
        for face in mesh.faces:
            pts3d = np.array([verts3d[i][:3] for i in face])
            v1, v2 = pts3d[1]-pts3d[0], pts3d[2]-pts3d[0]
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal)
            view_dir = -pts3d[0]  # câmera no (0,0,0)
            if np.dot(normal, view_dir) <= 0:  continue  # face não visível
            visible_faces.append(face)

        for face in visible_faces:

            pts = np.array([verts2d[i] for i in face])
            
            # verifica se o vertice está fora da tela
            if MatrixOperations.is_out_of_bounds(pts, self.width, self.height): continue
            
            if render_type == 'wireframe':
                pg.draw.polygon(self.screen, pg.Color('orange'), pts, 1)
            
            elif render_type == 'solid':
                pg.draw.polygon(self.screen, pg.Color('orange'), pts, 0)

            elif render_type == 'solid|shader':
                color = self.shader.shade(verts3d, face) if self.shader else pg.Color('orange')
                pg.draw.polygon(self.screen, color, pts, 0)
            
            elif render_type == 'textured' and self.texture is not None:
                gfx.textured_polygon(self.screen, pts.astype(int), self.texture, 0, 0)

class App:

    def __init__(self, scene, fps=30, width=1600, height=900):
        pg.init()
        self.fps = fps
        self.scene = scene
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((width,height))
        self.camera = Camera(width, height, position=np.array([0, 0, -9, 1]))
        texture = pg.image.load('./assets/textures/gold.png').convert()
        self.renderer = Renderer(self.screen, width, height, texture=texture, shader=LambertShader())

    def run(self):

        angle = 0.01
        rot_x = MatrixOperations.rotate_pitch(angle)
        rot_y = MatrixOperations.rotate_yaw(angle)
        rot_z = MatrixOperations.rotate_roll(angle)

        while True:

            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type==pg.KEYDOWN and e.key==pg.K_ESCAPE): pg.quit(); exit()
            
            self.screen.fill(pg.Color('darkslategray'))
            
            for mesh in self.scene:
                # mesh.apply_transform(np.eye(4))                               # placeholder
                mesh.apply_transform(rot_y @ rot_x @ rot_z)                     # auto-rotation
                self.renderer.render(mesh, self.camera, render_type='solid|shader')    # 'wireframe', 'solid', 'solid|shader', 'textured'
            
            pg.display.flip()
            self.clock.tick(self.fps)


if __name__=='__main__':

    faces, verts = FileManager('./assets/models/box/model.obj').load()

    #                                      x     y     z     w
    # mesh0 = Mesh(faces, verts, position=[  0.0, -1.5, -5.0,  1.0 ])
    # mesh1 = Mesh(faces, verts, position=[  0.0,  1.5, -5.0,  1.0 ])

    # scene = Scene()
    # scene.add(mesh0)
    # scene.add(mesh1)

    scene = Scene()
    scene.add(Mesh(faces, verts, position=[ 0.0, 0.0, 0.0, 1.0 ]))
    
    App(scene).run()
