
import numpy as np                                                              # type: ignore
import pygame as pg                                                             # type: ignore
from camera import Camera
from renderer import Renderer
from shader import LambertShader


class App:

    def __init__(self, scene, fps=30, width=1600, height=900, clock=None, screen=None, render_type='wireframe'):

        self.fps, self.scene, self.clock, self.screen, = fps, scene, clock, screen
        self.camera = Camera(width, height, position=np.array([0.0, 0.0, -9.0, 1.0]))        
        self.renderer = Renderer(self.screen, width, height, render_type, shader=LambertShader())

    def run(self):

        while True:

            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type==pg.KEYDOWN and e.key==pg.K_ESCAPE): pg.quit(); exit()
            
            self.screen.fill(pg.Color('darkslategray'))
            
            for mesh in self.scene:
                # mesh.apply_transform(np.eye(4))                               # placeholder
                mesh.auto_rotate(0.01)
                self.renderer.render(self.camera, mesh) 
            
            pg.display.flip()
            self.clock.tick(self.fps)
