# Import required libraries
from Polygon import *

class FileManager:

    def __init__(self, filepath):
        self.filepath = filepath
        self.faces:    list[list[int]]   = []
        self.vertices: list[list[float]] = []

    def load(self) -> 'FileManager':
        with open(self.filepath) as f:
            for line in f:
                if line.startswith('v '):  # Parse vertices
                    self.vertices.append([float(i) for i in line.split()[1:]] + [1])
                elif line.startswith('f'):  # Parse faces
                    self.faces.append([int(f.split('/')[0]) - 1 for f in line.split()[1:]])
        return self

    def get_faces(self) -> list[list[int]]:
        return self.faces
    
    def get_vertices(self) -> list[list[float]]:
        return self.vertices

    def get_dict(self) -> dict:
        return { 'faces': self.faces, 'vertices': self.vertices }
    
    def get_polygon(self) -> Polygon:
        return Polygon(self.faces, self.vertices)
