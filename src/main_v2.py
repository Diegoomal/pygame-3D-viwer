# Import required libraries
from engine import *

class App:

    # def __init__(self, polygon:Polygon, fps:int=30, width:int=1600, height:int=900):

    #     pg.init()
    #     self.fps = fps
    #     self.clock = pg.time.Clock()
    #     self.screen = pg.display.set_mode((width, height))

    #     self.polygon: Polygon = polygon
    #     self.polygon.vertices = self.polygon.vertices @ MatrixOperations.scale(0.95)

    #     self.render: Render = Render(width=width, height=height)
    #     self.camera: Camera = Camera(width=width, height=height, position=np.array([0, 0, -5, 1]))

    def __init__(self, polygon: Polygon, texture_path: str, fps=30, width=1600, height=900):
        pg.init()
        self.fps = fps
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((width, height))
        self.polygon = polygon
        self.polygon.vertices = self.polygon.vertices @ MatrixOperations.scale(0.95)
        self.render = Render(width, height)
        self.camera = Camera(width=width, height=height)
        self.texture = pg.image.load(texture_path).convert()

    def update(self):

        # INPUT HANDLER
        for event in pg.event.get(): 
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE): pg.quit(); exit()

        self.polygon.update()
        self.polygon.process(self.camera)
        
    # def draw(self):
    #     self.screen.fill( pg.Color('darkslategray') )
    #     self.render.polygon_to_screen(self.screen, self.polygon)
    #     # Update the display and maintain frame rate
    #     pg.display.set_caption(f"FPS: {self.clock.get_fps():.2f}")
    #     pg.display.flip()
    #     self.clock.tick(self.fps)

    def draw(self):
        self.screen.fill(pg.Color('black'))
        self.render.polygon_to_screen(self.screen, self.polygon, self.texture)
        pg.display.set_caption(f"FPS: {self.clock.get_fps():.2f}")
        pg.display.flip()
        self.clock.tick(self.fps)

    def run(self):
        while True:
            self.update()
            self.draw()

# Entry point
if __name__ == '__main__':
    polygon: Polygon = FileManager('./models/suzanne/model.obj').load().get_polygon()
    App(polygon=polygon, texture_path='./models/suzanne/texture.png').run()