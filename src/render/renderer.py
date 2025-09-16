import numpy as np                                                              # type: ignore
import pygame as pg                                                             # type: ignore
import pygame.gfxdraw as gfx                                                    # type: ignore
from core.matrix_operations import MatrixOperations
from render.rasterizer import Rasterizer


class Renderer:

    def __init__(self, screen, width, height, render_type='wireframe', shader=None):
        self.screen = screen
        self.width = width
        self.height = height
        self.render_type = render_type
        self.shader = shader
        self.zbuffer = None

    def _calculate_projection(self, camera, vertex):
    
        # vertex: Nx4 (model/world space)
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()
        screen_m = camera.get_screen_matrix()

        vertex_3d = vertex @ view @ proj                                        # clip-space (x,y,z,w)
        w = vertex_3d[:, 3].copy()
        w[np.isclose(w, 0.0)] = 1e-6                                            # evita div/0
        vertex_3d_norm = vertex_3d / w.reshape(-1, 1)                           # normalized device coords
        screen = vertex_3d_norm @ screen_m
        vertex_2d = screen[:, :2].copy()
        return vertex_2d, vertex_3d, vertex_3d_norm

    def render(self, camera, mesh):

        vertex_world = mesh.get_transformed()
        vertex_2d, vertex_3d, vertex_3d_norm = self._calculate_projection(camera, vertex_world)

        # z-buffer orientado como surfarray (x, y)
        self.zbuffer = np.full((self.width, self.height), np.inf, dtype=np.float32)

        sorted_faces = sorted(
            mesh.faces,
            key=lambda f: np.mean([vertex_3d_norm[i][2] for i in f]),
            reverse=False
        )

        for face in sorted_faces:
            vertex_2d_pts = np.array([vertex_2d[i] for i in face])

            if MatrixOperations.is_out_of_bounds(vertex_2d_pts, self.width, self.height):
                continue

            if self.render_type == 'wireframe':
                pg.draw.polygon(self.screen, pg.Color('orange'), vertex_2d_pts, 1)

            elif self.render_type == 'solid':
                pg.draw.polygon(self.screen, pg.Color('orange'), vertex_2d_pts, 0)

            elif self.render_type == 'solid|shader':
                color = self.shader.shade(face, vertex_world)
                pg.draw.polygon(self.screen, color, vertex_2d_pts, 0)

            elif self.render_type == 'textured' and mesh.texture is not None:
                gfx.textured_polygon(self.screen, vertex_2d_pts, mesh.texture, 0, 0)

            elif self.render_type == 'textured|rasterizer' and mesh.texture is not None:
                Rasterizer.draw_textured_triangle(self.screen, vertex_2d, vertex_3d, vertex_3d_norm, face, mesh.texture, self.zbuffer)
