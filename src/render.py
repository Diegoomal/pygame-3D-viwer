# Import required libraries
import numpy as np                                                              # type: ignore
import pygame as pg                                                             # type: ignore
import pygame.gfxdraw as gfx                                                    # type: ignore
from polygon import *


class Render:

    def __init__(self, width:int=1600, height:int=900, texture:pg.Surface|None=None):
        self.width = width
        self.height = height
        self.texture = texture

    def polygon_to_screen(self, screen, polygon:Polygon):

        if polygon is None: raise ValueError("Polygon cannot be None")

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
