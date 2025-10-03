class Scene:

    def __init__(self):
        self.meshes = []

    def add(self, mesh):
        self.meshes.append(mesh)

    def __iter__(self):
        return iter(self.meshes)
