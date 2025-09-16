
import pygame as pg                                                             # type: ignore
from app import App
from core.mesh import Mesh
from core.scene import Scene
from utils.file_manager import FileManager


if __name__=='__main__':

    pg.init()
    clock, screen = pg.time.Clock(), pg.display.set_mode((1600, 900))

    faces, verts = FileManager('./assets/models/box/model1.obj').load()
    texture = pg.image.load('./assets/textures/gold.png').convert()

    scene = Scene()
    scene.add(
        Mesh(faces, verts, position=[ 0.0, 0.0, 0.0, 1.0 ], texture=texture)
    )
    
    # 'wireframe', 'solid', 'solid|shader', 'textured', 'textured|rasterizer'
    App(scene, clock=clock, screen=screen, render_type='textured|rasterizer').run()
