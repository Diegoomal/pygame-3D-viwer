
import argparse
import pygame as pg                                                             # type: ignore
from app import App
from core.mesh import Mesh
from core.scene import Scene
from utils.file_manager import FileManager


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--render_type", type=str, default='wireframe')
    parser.add_argument("--model_name", type=str, default='./assets/models/box3.obj')
    parser.add_argument("--texture_name", type=str, default='./assets/textures/gold.png')
    return parser.parse_args()


def main():

    args = get_arguments()

    pg.init()
    clock, screen = pg.time.Clock(), pg.display.set_mode((args.width, args.height))

    faces, verts = FileManager(args.model_name).load()

    texture = None
    try:
        texture = pg.image.load(args.texture_name).convert()
    except Exception:
        texture = None

    scene = Scene()
    scene.add(
        Mesh(faces, verts, position=[ 0.0, 0.0, 0.0, 1.0 ], texture=texture)
    )
    
    # 'wireframe', 'solid', 'solid|shader', 'textured', 'textured|rasterizer'
    App(scene, clock=clock, screen=screen, render_type=args.render_type).run()


if __name__=='__main__':
    main()
