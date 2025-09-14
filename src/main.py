import math
import numpy as np                      # type: ignore
import pygame as pg                     # type: ignore
import pygame.gfxdraw as gfx            # type: ignore
from matrix_operations import MatrixOperations


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

    def shade(self, face, vertex_3d):
        
        vertex_3d = vertex_3d[:, :3]                                            # x,y,z,w -> drop w component

        vertex_3d_pts = np.array([vertex_3d[i] for i in face])
        
        v1 = vertex_3d_pts[1] - vertex_3d_pts[0]
        v2 = vertex_3d_pts[2] - vertex_3d_pts[0]
        
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        
        intensity = max(0, np.dot(normal, self.light_dir))
        
        val = int(255 * intensity)

        color = (val, val, val)                                                   # R, G, B

        return color

class Renderer:

    def __init__(self, screen, width, height, shader=None):
        self.screen = screen
        self.width = width
        self.height = height
        self.shader = shader

    def _calculate_projection(self, camera, vertex):

        vertex_3d = vertex.copy()

        _vertex = vertex_3d @ camera.get_view_matrix() @ camera.get_projection_matrix()
        _vertex /= _vertex[:, -1].reshape(-1, 1)
        _vertex = _vertex @ camera.get_screen_matrix()
        
        vertex_2d = _vertex[:, :2].copy()

        # print(
        #     "len(vertex):", len(vertex_2d), 
        #     "vertex_2d (x,y):", vertex_2d[0],
        #     "vertex_3d (x,y,z,w):", vertex_3d[0]
        # )

        return vertex_2d, vertex_3d

    def _draw_textured_triangle(self, surface, pts2d, ptsuv, texture):
        # pts2d: [(x,y), (x,y), (x,y)] em tela
        # ptsuv: [(u,v), (u,v), (u,v)] entre 0..1
        tex_w, tex_h = texture.get_size()
        tex_pixels = pg.surfarray.pixels3d(texture)
        min_x = max(int(min(p[0] for p in pts2d)), 0)
        max_x = min(int(max(p[0] for p in pts2d)), surface.get_width()-1)
        min_y = max(int(min(p[1] for p in pts2d)), 0)
        max_y = min(int(max(p[1] for p in pts2d)), surface.get_height()-1)

        def edge(p1, p2, p):
            return (p[0]-p1[0])*(p2[1]-p1[1]) - (p[1]-p1[1])*(p2[0]-p1[0])

        area = edge(pts2d[0], pts2d[1], pts2d[2])

        surf_array = pg.surfarray.pixels3d(surface)
        for y in range(min_y, max_y+1):
            for x in range(min_x, max_x+1):
                p = (x+0.5, y+0.5)
                w0 = edge(pts2d[1], pts2d[2], p)
                w1 = edge(pts2d[2], pts2d[0], p)
                w2 = edge(pts2d[0], pts2d[1], p)
                if w0>=0 and w1>=0 and w2>=0:
                    b0, b1, b2 = w0/area, w1/area, w2/area
                    u = b0*ptsuv[0][0] + b1*ptsuv[1][0] + b2*ptsuv[2][0]
                    v = b0*ptsuv[0][1] + b1*ptsuv[1][1] + b2*ptsuv[2][1]
                    tx = int(u*tex_w) % tex_w
                    ty = int(v*tex_h) % tex_h
                    surf_array[x, y] = tex_pixels[tx, ty]
        del surf_array
        del tex_pixels

    def render(self, camera, mesh, render_type='wireframe'):

        vertex_2d, vertex_3d = self._calculate_projection(camera, mesh.get_transformed())

        sorted_faces = sorted(
            mesh.faces,
            key=lambda f: np.mean([vertex_3d[i][2] for i in f]),
            reverse=False
        )

        for face in sorted_faces:

            # recupera os pontos (x,y) da face atual
            vertex_2d_pts = np.array([vertex_2d[i] for i in face])
            
            # verifica se o vertice está fora da tela
            if MatrixOperations.is_out_of_bounds(vertex_2d_pts, self.width, self.height): continue
            
            if render_type == 'wireframe':
                pg.draw.polygon(self.screen, pg.Color('orange'), vertex_2d_pts, 1)
            
            elif render_type == 'solid':
                pg.draw.polygon(self.screen, pg.Color('orange'), vertex_2d_pts, 0)

            elif render_type == 'solid|shader':
                color = self.shader.shade(face, vertex_3d)
                pg.draw.polygon(self.screen, color, vertex_2d_pts, 0)

            elif render_type == 'textured' and mesh.texture is not None:
                gfx.textured_polygon(self.screen, vertex_2d_pts, mesh.texture, 0, 0)

            elif render_type == 'textured|mapping' and mesh.texture is not None:
                # exemplo UV automático por posição 3D (ajuste conforme seu .obj tiver UVs)
                pts2d = [tuple(vertex_2d[i]) for i in face[:3]]
                ptsuv = [(vertex_3d[i][0]%1, vertex_3d[i][1]%1) for i in face[:3]]
                self._draw_textured_triangle(self.screen, pts2d, ptsuv, mesh.texture)

class App:

    def __init__(self, scene, fps=30, width=1600, height=900, clock=None, screen=None, render_type='wireframe'):

        self.fps, self.scene, self.clock, self.screen, self.render_type = fps, scene, clock, screen, render_type
        self.camera = Camera(width, height, position=np.array([0.0, 0.0, -9.0, 1.0]))        
        self.renderer = Renderer(self.screen, width, height, shader=LambertShader())

    def run(self):

        while True:

            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type==pg.KEYDOWN and e.key==pg.K_ESCAPE): pg.quit(); exit()
            
            self.screen.fill(pg.Color('darkslategray'))
            
            for mesh in self.scene:
                # mesh.apply_transform(np.eye(4))                               # placeholder
                mesh.auto_rotate(0.01)
                self.renderer.render(self.camera, mesh, render_type=self.render_type) 
            
            pg.display.flip()
            self.clock.tick(self.fps)


if __name__=='__main__':

    pg.init()
    clock, screen = pg.time.Clock(), pg.display.set_mode((1600, 900))
    
    file_path = './assets/models/box/model1.obj'
    # file_path = './assets/models/suzanne/model.obj'
    faces, verts = FileManager(file_path).load()

    scene = Scene()
    scene.add(
        Mesh(
            faces, verts, position=[ 0.0, 0.0, 0.0, 1.0 ],
            texture=pg.image.load('./assets/textures/gold.png').convert()
        )
    )
    
    # 'wireframe', 'solid', 'solid|shader', 'textured', 'textured|mapping'
    App(scene, clock=clock, screen=screen, render_type='textured|mapping').run()
