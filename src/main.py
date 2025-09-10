# Import required libraries
from file_manager import *
from camera import *
from polygon import *
from render import *


class App:

    def __init__(self, polygon:Polygon, fps:int=30, width:int=1600, height:int=900):

        pg.init()
        self.fps = fps
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((width, height))

        self.polygon: Polygon = polygon
        self.camera: Camera = Camera(
            width=width, 
            height=height, 
            position=np.array([0, 0, -5, 1])
        )

        # wired
        self.render: Render = Render(width=width, height=height)

        # textured
        # texture = pg.image.load('./assets/textures/gold.png').convert()
        # self.render: Render = Render(width=width, height=height, texture=texture)

    def update(self):

        # INPUT HANDLER
        for event in pg.event.get(): 
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE): pg.quit(); exit()

        self.polygon.update()
        self.polygon.process(self.camera)
    
    def draw(self):
        self.screen.fill( pg.Color('darkslategray') )
        self.render.polygon_to_screen(self.screen, self.polygon)
        # Update the display and maintain frame rate
        pg.display.set_caption(f"FPS: {self.clock.get_fps():.2f}")
        pg.display.flip()
        self.clock.tick(self.fps)

    def run(self):
        while True:
            self.update()
            self.draw()

# Entry point
if __name__ == '__main__':
    
    polygon: Polygon = FileManager('./assets/models/suzanne/model.obj').load().get_polygon()
    
    App(polygon=polygon).run()
