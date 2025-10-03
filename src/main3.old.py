"""
Refatoração do visualizador 3D com separação de responsabilidades.
Arquivo único contendo: utilitários de matriz, Transform, Mesh, Scene,
FileManager, Camera, Shader, Rasterizer, Renderer, App.
"""
import math
import numpy as np                                                              # type: ignore
import pygame as pg                                                             # type: ignore
import pygame.gfxdraw as gfx                                                    # type: ignore

# -------------------------
# Matrix utilities
# -------------------------
class MatrixOps:
    @staticmethod
    def translate(x, y, z):
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [  x,   y,   z, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def scale(sx, sy=None, sz=None):
        if sy is None: sy = sx
        if sz is None: sz = sx
        return np.array([
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def rotate_x(a):
        s, c = math.sin(a), math.cos(a)
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0,   c,   s, 0.0],
            [0.0,  -s,   c, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def rotate_y(a):
        s, c = math.sin(a), math.cos(a)
        return np.array([
            [  c, 0.0,  -s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [  s, 0.0,   c, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def rotate_z(a):
        s, c = math.sin(a), math.cos(a)
        return np.array([
            [  c,   s, 0.0, 0.0],
            [ -s,   c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def is_out_of_bounds(arr, width, height):
        # arr: Nx2 array of screen coords
        if arr.size == 0: return True
        x = arr[:, 0]
        y = arr[:, 1]
        return np.any((x < 0) | (x > width) | (y < 0) | (y > height))


# -------------------------
# Transform (position, rotation, scale)
# -------------------------
class Transform:

    def __init__(self, position=None, rotation=None, scale=None):
        self.position   = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        # rotation as Euler angles (rx, ry, rz)
        self.rotation   = np.array(rotation if rotation is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.scale      = np.array(scale    if scale    is not None else [1.0, 1.0, 1.0], dtype=np.float64)

    def matrix(self):
        t = MatrixOps.translate(*self.position)
        rx = MatrixOps.rotate_x(self.rotation[0])
        ry = MatrixOps.rotate_y(self.rotation[1])
        rz = MatrixOps.rotate_z(self.rotation[2])
        s = MatrixOps.scale(*self.scale)
        # Order: scale, rotateZ * rotateY * rotateX, translate
        return s @ (rz @ (ry @ rx)) @ t


# -------------------------
# Mesh (data) with Transform
# -------------------------
class Mesh:

    def __init__(self, faces, vertices, uvs=None, normals=None, transform=None, texture=None):
        # faces: list[list[int]] (triangles assumed). vertices: list[list[float]] size Nx3 or Nx4
        self.faces = np.array([np.array(f, dtype=int) for f in faces], dtype=object)
        self.vertices = np.array(vertices, dtype=float)
        if self.vertices.shape[1] == 3:
            # append w=1
            self.vertices = np.hstack((self.vertices, np.ones((self.vertices.shape[0], 1), dtype=float)))
        self.uvs = np.array(uvs, dtype=float) if uvs is not None else None
        self.normals = np.array(normals, dtype=float) if normals is not None else None
        self.transform = transform if transform is not None else Transform()
        self.texture = texture

    def world_vertices(self):
        M = self.transform.matrix()
        # vertices is Nx4, multiply each row by M
        return (self.vertices @ M)

    def apply_transform_matrix(self, mat):
        # compose a matrix into transform by updating vertex positions in model space (optional)
        # keep transform intact; more explicit transforms should be done via Transform object
        self.vertices = (self.vertices @ mat)


# -------------------------
# Scene (container)
# -------------------------
class Scene:

    def __init__(self):
        self.meshes = []
        self.lights = []                                                        # list of light dicts {'pos': np.array, 'color': (r,g,b)}
        self.cameras = []

    def add_mesh(self, mesh):
        self.meshes.append(mesh)

    def add_light(self, pos, color=(1.0, 1.0, 1.0)):
        self.lights.append({'pos': np.array(pos, dtype=float), 'color': color})

    def add_camera(self, cam):
        self.cameras.append(cam)

    def __iter__(self):
        return iter(self.meshes)


# -------------------------
# FileManager (OBJ minimal)
# -------------------------
class FileManager:

    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        faces, verts, uvs, norms = [], [], [], []
        with open(self.filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('vt '):
                    parts = line.split()
                    uvs.append([float(parts[1]), float(parts[2])])
                elif line.startswith('vn '):
                    parts = line.split()
                    norms.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    face_idx = []
                    for p in parts:
                        # support v/vt/vn, v//vn, v/vt
                        comp = p.split('/')
                        v = int(comp[0]) - 1
                        face_idx.append(v)
                    # triangulate if needed (fan)
                    if len(face_idx) == 3:
                        faces.append(face_idx)
                    elif len(face_idx) > 3:
                        for i in range(1, len(face_idx)-1):
                            faces.append([face_idx[0], face_idx[i], face_idx[i+1]])
        return faces, verts, uvs if uvs else None, norms if norms else None


# -------------------------
# Camera (transform + projection)
# -------------------------
class Camera:

    def __init__(self, width, height, position=None, rotation=None, h_fov=math.pi/3, near=0.1, far=100.0):
        self.width = width
        self.height = height
        self.transform = Transform(position=position or [0.0, 0.0, -5.0], rotation=rotation or [0.0, 0.0, 0.0])
        self.h_fov = h_fov
        self.v_fov = h_fov * (height / width)
        self.near = near
        self.far = far

    def view_matrix(self):
        # view matrix is inverse of camera world matrix
        M = self.transform.matrix()
        return np.linalg.inv(M)

    def projection_matrix(self):
        m00 = 1.0 / math.tan(self.h_fov / 2.0)
        m11 = 1.0 / math.tan(self.v_fov / 2.0)
        m22 = -(self.far + self.near) / (self.far - self.near)
        m32 = -2.0 * self.near * self.far / (self.far - self.near)
        return np.array([
            [m00, 0.0, 0.0, 0.0],
            [0.0, m11, 0.0, 0.0],
            [0.0, 0.0, m22, -1.0],
            [0.0, 0.0, m32, 0.0]
        ], dtype=np.float64)

    def screen_matrix(self):
        hw, hh = self.width // 2, self.height // 2
        return np.array([
            [hw,  0.0, 0.0, 0.0],
            [0.0, -hh, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [hw, hh, 0.0, 1.0]
        ], dtype=np.float64)


# -------------------------
# Shader (Lambert)
# -------------------------
class LambertShader:

    def __init__(self, light_dir=np.array([0, 0, -1], dtype=float)):
        d = np.array(light_dir, dtype=float)
        n = np.linalg.norm(d)
        self.light_dir = d / (n if n != 0 else 1.0)

    def shade_face(self, face, vertex_world):
        # vertex_world: Nx4 array of mesh world vertices
        # compute normal from first triangle vertices
        pts = np.array([vertex_world[i][:3] for i in face[:3]])
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0: return (0, 0, 0)
        normal /= norm
        intensity = max(0.0, np.dot(normal, self.light_dir))
        val = int(255 * intensity)
        return (val, val, val)


# -------------------------
# Rasterizer (textured triangle)
# -------------------------
class Rasterizer:

    @staticmethod
    def draw_textured_triangle(surface: pg.Surface, vertex_2d, vertex_3d, face, texture: pg.Surface):

        pts2d = [tuple(vertex_2d[i]) for i in face[:3]]
        ptsuv = [(vertex_3d[i][0] % 1.0, vertex_3d[i][1] % 1.0) for i in face[:3]]

        tex_w, tex_h = texture.get_size()
        tex_pixels = pg.surfarray.pixels3d(texture)

        min_x, max_x = max(int(min(p[0] for p in pts2d)), 0), min(int(max(p[0] for p in pts2d)), surface.get_width() - 1)
        min_y, max_y = max(int(min(p[1] for p in pts2d)), 0), min(int(max(p[1] for p in pts2d)), surface.get_height() - 1)

        def edge(p1, p2, p): return (p[0] - p1[0]) * (p2[1] - p1[1]) - (p[1] - p1[1]) * (p2[0] - p1[0])

        area = edge(pts2d[0], pts2d[1], pts2d[2])
        if area == 0: return
        surf_array = pg.surfarray.pixels3d(surface)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = (x + 0.5, y + 0.5)
                w0 = edge(pts2d[1], pts2d[2], p)
                w1 = edge(pts2d[2], pts2d[0], p)
                w2 = edge(pts2d[0], pts2d[1], p)
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    b0, b1, b2 = w0 / area, w1 / area, w2 / area
                    u = b0 * ptsuv[0][0] + b1 * ptsuv[1][0] + b2 * ptsuv[2][0]
                    v = b0 * ptsuv[0][1] + b1 * ptsuv[1][1] + b2 * ptsuv[2][1]
                    tx = int(u * tex_w) % tex_w
                    ty = int(v * tex_h) % tex_h
                    surf_array[x, y] = tex_pixels[tx, ty]

        del surf_array
        del tex_pixels


# -------------------------
# Renderer (pipeline: projection, culling, shading, raster)
# -------------------------
class Renderer:

    def __init__(self, screen, width, height, render_type='wireframe', shader=None):
        self.screen = screen
        self.width = width
        self.height = height
        self.render_type = render_type
        self.shader = shader if shader is not None else LambertShader()

    def project(self, camera: Camera, vertices_world: np.ndarray):
        # vertices_world: Nx4
        V = camera.view_matrix()
        P = camera.projection_matrix()
        S = camera.screen_matrix()

        # apply view and projection
        clip = (vertices_world @ V) @ P  # Nx4
        # perspective divide
        w = clip[:, 3].reshape(-1, 1)
        # avoid divide by zero
        w[w == 0] = 1e-6
        ndc = clip / w
        screen = ndc @ S
        screen2d = screen[:, :2].copy()
        return screen2d, vertices_world

    def render_mesh(self, camera: Camera, mesh: Mesh):
        verts_world = mesh.world_vertices()
        verts_2d, verts_world = self.project(camera, verts_world)

        # depth sorting by face centroid z in world space (farther z greater)
        sorted_faces = sorted(mesh.faces, key=lambda f: np.mean([verts_world[i][2] for i in f]), reverse=True)

        for face in sorted_faces:
            face = np.array(face, dtype=int)
            v2d_pts = np.array([verts_2d[i] for i in face])
            if MatrixOps.is_out_of_bounds(v2d_pts, self.width, self.height):
                continue

            if self.render_type == 'wireframe':
                pg.draw.polygon(self.screen, pg.Color('orange'), v2d_pts, 1)

            elif self.render_type == 'solid':
                pg.draw.polygon(self.screen, pg.Color('orange'), v2d_pts, 0)

            elif self.render_type == 'solid_shader':
                color = self.shader.shade_face(face, verts_world)
                pg.draw.polygon(self.screen, color, v2d_pts, 0)

            elif self.render_type == 'textured' and mesh.texture is not None:
                # fallback textured polygon using gfx (may be slow and limited)
                gfx.textured_polygon(self.screen, v2d_pts, mesh.texture, 0, 0)

            elif self.render_type == 'textured_rasterizer' and mesh.texture is not None:
                Rasterizer.draw_textured_triangle(self.screen, verts_2d, verts_world, face, mesh.texture)

    def render(self, camera: Camera, scene: Scene):
        for mesh in scene:
            self.render_mesh(camera, mesh)


# -------------------------
# App (engine loop)
# -------------------------
class App:

    def __init__(self, scene, fps=60, width=1600, height=900, clock=None, screen=None, render_type='wireframe'):
        self.fps = fps
        self.scene = scene
        self.clock = clock
        self.screen = screen
        self.camera = Camera(width, height, position=[0.0, 0.0, -9.0])
        self.renderer = Renderer(self.screen, width, height, render_type, shader=LambertShader())

    def run(self):

        while True:
            
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    pg.quit()
                    exit()

            self.screen.fill(pg.Color('darkslategray'))

            for mesh in self.scene:
                # minimal animation separated from transform internals
                mesh.transform.rotation[1] += 0.01
                self.renderer.render_mesh(self.camera, mesh)

            pg.display.flip()
            if self.clock:
                self.clock.tick(self.fps)


# -------------------------
# Entrypoint (exemplo)
# -------------------------

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--render_type", type=str, default='wireframe')
    parser.add_argument("--model_name", type=str, default='./assets/models/box/model1.obj')
    parser.add_argument("--texture_name", type=str, default='./assets/textures/gold.png')
    
    args = parser.parse_args()

    pg.init()
    clock = pg.time.Clock()
    screen = pg.display.set_mode((args.width, args.height))

    faces, verts, uvs, norms = FileManager(args.model_name).load()

    tex = None
    try:
        tex = pg.image.load(args.texture_name).convert()
    except Exception:
        tex = None

    scene = Scene()
    scene.add_mesh(Mesh(faces, verts, uvs, norms, transform=Transform(position=[0.0, 0.0, 0.0]), texture=tex))

    App(scene, fps=60, width=args.width, height=args.height, clock=clock, screen=screen, render_type=args.render_type).run()


if __name__ == '__main__':
    main()


# python src/main3.old.py --width 1600 --height 900 --render_type wireframe --model_name ./assets/models/box2.obj
# python src/main3.old.py --width 1600 --height 900 --render_type textured --model_name ./assets/models/box2.obj --texture_name ./assets/textures/gold.png
# python src/main3.old.py --width 1600 --height 900 --render_type solid_shader --model_name ./assets/models/box2.obj
# python src/main3.old.py --width 1600 --height 900 --render_type textured_rasterizer --model_name ./assets/models/box2.obj --texture_name ./assets/textures/gold.png