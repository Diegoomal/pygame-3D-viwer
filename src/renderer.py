import numpy as np                                                              # type: ignore
import pygame as pg                                                             # type: ignore
import pygame.gfxdraw as gfx                                                    # type: ignore
from matrix_operations import MatrixOperations
from rasterizer import Rasterizer

class Renderer:

    def __init__(self, screen, width, height, render_type='wireframe', shader=None):
        self.screen = screen
        self.width = width
        self.height = height
        self.render_type = render_type
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

    def render(self, camera, mesh):

        vertex_2d, vertex_3d = self._calculate_projection(camera, mesh.get_transformed())

        sorted_faces = sorted(mesh.faces, key=lambda f: np.mean([vertex_3d[i][2] for i in f]), reverse=False)

        for face in sorted_faces:

            # recupera os pontos (x,y) da face atual
            vertex_2d_pts = np.array([vertex_2d[i] for i in face])

            # verifica se o vertice está fora da tela
            if MatrixOperations.is_out_of_bounds(vertex_2d_pts, self.width, self.height): continue
            
            if self.render_type == 'wireframe':
                pg.draw.polygon(self.screen, pg.Color('orange'), vertex_2d_pts, 1)
            
            elif self.render_type == 'solid':
                pg.draw.polygon(self.screen, pg.Color('orange'), vertex_2d_pts, 0)

            elif self.render_type == 'solid|shader':
                color = self.shader.shade(face, vertex_3d)
                pg.draw.polygon(self.screen, color, vertex_2d_pts, 0)

            elif self.render_type == 'textured' and mesh.texture is not None:
                gfx.textured_polygon(self.screen, vertex_2d_pts, mesh.texture, 0, 0)

            elif self.render_type == 'textured|uv_mapping' and mesh.texture is not None:
                # exemplo UV automático por posição 3D (ajuste conforme seu .obj tiver UVs)
                Rasterizer.draw_textured_triangle(self.screen, vertex_2d, vertex_3d, face, mesh.texture)
