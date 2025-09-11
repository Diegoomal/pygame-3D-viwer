import numpy as np                                                              # type: ignore
import pygame as pg                                                             # type: ignore
import pygame.gfxdraw as gfx                                                    # type: ignore
from core.polygon import *


class Renderer:

    def __init__(self, screen, width:int=1600, height:int=900, texture:pg.Surface|None=None):
        self.screen = screen
        self.width = width
        self.height = height
        self.texture = texture

    def render(self, polygon:Polygon):

        if polygon is None: raise ValueError("Polygon cannot be None")

        faces, tvs2d = polygon.faces, polygon.transformed_vertices_2d

        for face in faces:
            pts = np.array([tvs2d[i] for i in face])
            if not MatrixOperations.is_out_of_bounds(pts, self.width, self.height):
                if self.texture is None:
                    # wireframe
                    pg.draw.polygon(self.screen, pg.Color('orange'), pts, 1)
                else:
                    # textured (sem UV → só mapeia textura inteira no polígono)
                    gfx.textured_polygon(self.screen, pts.astype(int), self.texture, 0, 0)

    # SIMPLE SHADER APPLYED
    # def polygon_to_screen(self, screen, polygon:Polygon):

    #     if polygon is None: raise ValueError("Polygon cannot be None")

    #     faces, tvs2d, tvs3d = polygon.faces, polygon.transformed_vertices_2d, polygon.transformed_vertices_3d

    #     light_dir = np.array([0, 0, -1])                                        # luz frontal

    #     for face in faces:

    #         pts2d = np.array([tvs2d[i] for i in face])
    #         pts3d = np.array([tvs3d[i][:3] for i in face])

    #         # normal da face
    #         v1, v2 = pts3d[1] - pts3d[0], pts3d[2] - pts3d[0]
    #         normal = np.cross(v1, v2)
    #         normal = normal / np.linalg.norm(normal)

    #         intensity = np.dot(normal, light_dir)
    #         if intensity > 0:                                                   # desenha só se estiver voltada para a luz
    #             color_val = int(255 * intensity)
    #             color = (color_val, color_val, 0)                               # tom amarelado
    #             pg.draw.polygon(screen, color, pts2d, 0)                        # preenchido
