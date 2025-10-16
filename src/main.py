#
# Simple 3D Viewer with Pygame
# filename: "main.py"
#

import math
import argparse
from abc import ABC, abstractmethod

import numpy as np                                                              # type: ignore
import pygame as pg                                                             # type: ignore
import pygame.gfxdraw as gfx                                                    # type: ignore

#
# Shader
#

class LambertShader:

    def __init__(self, light_dir=np.array([0, 0, -1])):
        self.light_dir = light_dir / np.linalg.norm(light_dir)

    def shade(self, face, vertex_3d):
        
        vertex_3d = vertex_3d[:, :3]                                            # x,y,z,w -> drop w component

        vertex_3d_pts = np.array([vertex_3d[i] for i in face])
        
        v1 = vertex_3d_pts[1] - vertex_3d_pts[0]
        v2 = vertex_3d_pts[2] - vertex_3d_pts[0]
        
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return (0, 0, 0)
        normal /= norm
        
        intensity = max(0, np.dot(normal, self.light_dir))
        
        val = int(255 * intensity)

        color = (val, val, val)                                                 # R, G, B

        return color


#
# Render
#

class Rasterizer:

    @staticmethod
    def draw_textured_triangle(surface: pg.Surface, vertex_2d, vertex_3d, vertex_3d_norm, face, texture: pg.Surface, zbuffer=None):

        # screen-space verts
        pts2d = [tuple(vertex_2d[i]) for i in face[:3]]

        # UVs derived from clip (u,v in 0..1)
        ptsuv = [((vertex_3d[i][0] / vertex_3d[i][3]) % 1.0, (vertex_3d[i][1] / vertex_3d[i][3]) % 1.0) for i in face[:3]]

        # clip-space w and ndc z (depth)
        ptsw = [vertex_3d[i][3] for i in face[:3]]
        ptsz = [vertex_3d_norm[i][2] for i in face[:3]]

        tex_w, tex_h = texture.get_size()
        tex_pixels = pg.surfarray.pixels3d(texture)
        surf_array = pg.surfarray.pixels3d(surface)

        min_x, max_x = max(int(min(p[0] for p in pts2d)), 0), min(int(max(p[0] for p in pts2d)), surface.get_width() - 1)
        min_y, max_y = max(int(min(p[1] for p in pts2d)), 0), min(int(max(p[1] for p in pts2d)), surface.get_height() - 1)

        def edge(p1, p2, p): return (p[0] - p1[0]) * (p2[1] - p1[1]) - (p[1] - p1[1]) * (p2[0] - p1[0])

        area = edge(pts2d[0], pts2d[1], pts2d[2])
        if abs(area) < 1e-9:
            del surf_array; del tex_pixels
            return

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):

                p = (x + 0.5, y + 0.5)
                w0, w1, w2 = edge(pts2d[1], pts2d[2], p), edge(pts2d[2], pts2d[0], p), edge(pts2d[0], pts2d[1], p)

                inside = (w0 >= 0 and w1 >= 0 and w2 >= 0) if area > 0 else (w0 <= 0 and w1 <= 0 and w2 <= 0)
                if not inside: continue

                b0, b1, b2 = w0 / area, w1 / area, w2 / area

                # perspective-correct interpolation of (u,v)
                inv_w = b0 / ptsw[0] + b1 / ptsw[1] + b2 / ptsw[2]
                if inv_w == 0: continue

                u = (b0 * ptsuv[0][0] / ptsw[0] + b1 * ptsuv[1][0] / ptsw[1] + b2 * ptsuv[2][0] / ptsw[2]) / inv_w
                v = (b0 * ptsuv[0][1] / ptsw[0] + b1 * ptsuv[1][1] / ptsw[1] + b2 * ptsuv[2][1] / ptsw[2]) / inv_w

                # depth from NDC z (lower typically means nearer)
                depth = b0 * ptsz[0] + b1 * ptsz[1] + b2 * ptsz[2]

                if zbuffer is None or depth < zbuffer[x, y]:
                    if zbuffer is not None:
                        zbuffer[x, y] = depth
                    tx = int(u * tex_w) % tex_w
                    ty = int(v * tex_h) % tex_h
                    surf_array[x, y] = tex_pixels[tx, ty]

        del surf_array; del tex_pixels


class Renderer:

    def __init__(self, screen, width, height, render_type='wireframe', shader=None):
        self.screen = screen
        self.width = width
        self.height = height
        self.render_type = render_type
        self.shader = shader
        self.zbuffer = None

    def render(self, camera, mesh):

        vertex_world = mesh.get_transformed()

        def calculate_projection(camera, vertex):
    
            # vertex: Nx4 (model/world space)
            view = camera.get_view_matrix()
            proj = camera.get_projection_matrix()
            screen_m = camera.get_screen_matrix()

            vertex_3d = vertex @ view @ proj                                    # clip-space (x,y,z,w)
            w = vertex_3d[:, 3].copy()
            w[np.isclose(w, 0.0)] = 1e-6                                        # evita div/0
            vertex_3d_norm = vertex_3d / w.reshape(-1, 1)                       # normalized device coords
            screen = vertex_3d_norm @ screen_m
            vertex_2d = screen[:, :2].copy()
            return vertex_2d, vertex_3d, vertex_3d_norm

        vertex_2d, vertex_3d, vertex_3d_norm = calculate_projection(camera, vertex_world)

        # z-buffer orientado como surfarray (x, y)
        self.zbuffer = np.full((self.width, self.height), np.inf, dtype=np.float32)

        sorted_faces = sorted(mesh.faces, key=lambda f: np.mean([vertex_3d_norm[i][2] for i in f]), reverse=False)

        for face in sorted_faces:
            vertex_2d_pts = np.array([vertex_2d[i] for i in face])

            if MatrixOperations.is_out_of_bounds(vertex_2d_pts, self.width, self.height): continue

            if self.render_type == 'wireframe':
                pg.draw.polygon(self.screen, pg.Color('orange'), vertex_2d_pts, 1)

            elif self.render_type == 'solid':
                pg.draw.polygon(self.screen, pg.Color('orange'), vertex_2d_pts, 0)

            elif self.render_type == 'solid|shader' and self.shader is not None:
                color = self.shader.shade(face, vertex_world)
                pg.draw.polygon(self.screen, color, vertex_2d_pts, 0)

            elif self.render_type == 'textured' and mesh.texture is not None:
                gfx.textured_polygon(self.screen, vertex_2d_pts, mesh.texture, 0, 0)

            elif self.render_type == 'textured|rasterizer' and mesh.texture is not None:
                Rasterizer.draw_textured_triangle(self.screen, vertex_2d, vertex_3d, vertex_3d_norm, face, mesh.texture, self.zbuffer)


#
# core
#

class MatrixOperations:
    
    # --- CHANGED: removed numba decorator to ensure runtime compatibility ---
    @staticmethod
    def is_out_of_bounds(arr, width, height):
        return np.any(
            (arr[:, 0] < 0)     | 
            (arr[:, 0] > width) | 
            (arr[:, 1] < 0)     | 
            (arr[:, 1] > height)
        )

    @staticmethod
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
    def scale(n):
        return np.array([
            [   n, 0.0, 0.0, 0.0 ],
            [ 0.0,   n, 0.0, 0.0 ],
            [ 0.0, 0.0,   n, 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0 ]
        ], dtype=np.float64)
   
    @staticmethod
    def rotate_pitch(angle):                                                    # X-axis
        s, c = np.sin(angle), np.cos(angle)
        return np.array([
            [ 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0,   c,   s, 0.0 ],
            [ 0.0,  -s,   c, 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0 ]
        ], dtype=np.float64)

    @staticmethod
    def rotate_yaw(angle):                                                      # Y-Axis
        s, c = np.sin(angle), np.cos(angle)
        return np.array([
            [   c, 0.0,   -s, 0.0 ],
            [ 0.0, 1.0,  0.0, 0.0 ],
            [   s, 0.0,    c, 0.0 ],
            [ 0.0, 0.0,  0.0, 1.0 ]
        ], dtype=np.float64)
    
    @staticmethod
    def rotate_roll(angle):                                                     # Z-Axis
        s, c = np.sin(angle), np.cos(angle)
        return np.array([
            [   c,   s, 0.0, 0.0 ],
            [  -s,   c, 0.0, 0.0 ],
            [ 0.0, 0.0, 1.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0 ]
        ], dtype=np.float64)


class Camera:

    def __init__(self, width, height, position=np.array([0,0,-5,1]), h_fov=math.pi/3, near=0.1, far=100):
        self.width, self.height = width, height
        self.position = position
        self.h_fov, self.v_fov = h_fov, h_fov*(height/width)
        self.near, self.far = near, far

    def get_view_matrix(self):
        x, y, z = self.position[:3]
        # note: we keep the same translate convention used by the renderer
        return MatrixOperations.translate(x, y, z)

    def get_projection_matrix(self):

        m00 = 1 / math.tan(self.h_fov / 2)
        m11 = 1 / math.tan(self.v_fov / 2)
        m22 = - (self.far + self.near) / (self.far - self.near)
        m32 = -2 * self.near * self.far / (self.far - self.near)
        
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


class Mesh:

    def __init__(self, faces:np.ndarray, vertices:np.ndarray, position=[0,0,0,1], texture_path=None):
        self.faces = np.array(faces, dtype=object)
        self.vertices = np.array(vertices, dtype=float)
        self.model_matrix = MatrixOperations.translate(*position[:3])
        self.texture = None
        self.texture_path = texture_path

    def load_texture(self):
        try:
            if self.texture_path is None:
                return None
            self.texture = pg.image.load(self.texture_path).convert_alpha()
        except Exception:
            self.texture = None
            return None

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

    def __init__(self, render_type='wireframe'):
        self.meshes = []
        self.render_type = render_type

    def load_textures(self):
        for mesh in self.meshes:
            if mesh.texture_path is not None:
                mesh.load_texture()

    def add(self, mesh):
        self.meshes.append(mesh)

    def __iter__(self):
        return iter(self.meshes)

#
# Utils
#

class FileManager:

    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):

        faces, vertices = [], []

        with open(self.filepath) as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(i) for i in parts[1:4]] + [1.0])
                elif line.startswith('f'):
                    faces.append([int(p.split('/')[0]) - 1 for p in line.split()[1:]])

        return faces, vertices

#
# Engine
#

class AbstractEngine(ABC):

    def __init__(self, width = 1600, height = 900, fps = 60):
        pg.init()
        self.fps, self.width, self.height = fps, width, height
        self.display = pg.display.set_mode((width, height))
        self.clock = pg.time.Clock()
        self.running = True

    def _on_update_start(self):
        self.on_handle_events()

    def _on_update_finish(self):
        self.clock.tick(self.fps)

    def _on_draw_start(self):
        self.display.fill((0, 0, 0))

    def _on_draw_finish(self):
        pg.display.flip()
        pg.display.set_caption(f"FPS: {self.clock.get_fps():.2f}")

    @abstractmethod
    def on_update(self, dt: float):
        raise NotImplementedError

    @abstractmethod
    def on_draw(self, surface: pg.Surface):
        raise NotImplementedError

    @abstractmethod
    def on_handle_events(self):
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                self.running = False

    def run(self):
        while self.running:
            self.on_handle_events()
            self._on_update_start()
            self.on_update(dt=self.clock.get_time() / 1000.0)
            self._on_draw_start()
            self.on_draw(self.display)
            self._on_draw_finish()
            self._on_update_finish()
        pg.quit()


class Engine(AbstractEngine):

    def __init__(self, scene: Scene):
        super().__init__()
        self.scene = scene
        self.scene.load_textures()
        self.camera = Camera(self.width, self.height, position=np.array([0.0, 0.0, -9.0, 1.0]))        
        self.renderer = Renderer(self.display, self.width, self.height, scene.render_type, shader=LambertShader())

    def on_handle_events(self):
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                self.running = False

    def on_update(self, dt: float):
        for mesh in self.scene:
            mesh.auto_rotate(0.01)

    def on_draw(self, surface: pg.Surface = None):
        # draw all meshes here so screen clearing happens before rendering
        for mesh in self.scene:
            self.renderer.render(self.camera, mesh)

#
# Main & args
#

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--render_type", type=str, default='wireframe')
    parser.add_argument("--model_name", type=str, default='./assets/models/box3.obj')
    parser.add_argument("--texture_name", type=str, default=None)
    return parser.parse_args()

def main():
    
    args = get_arguments()

    faces, verts = FileManager(args.model_name).load()

    scene = Scene(render_type=args.render_type)
    scene.add( Mesh(faces, verts, position=[ 0.0, 0.0, 0.0, 1.0 ]) )

    Engine(scene=scene).run()


if __name__ == "__main__":
    main()

# EOF