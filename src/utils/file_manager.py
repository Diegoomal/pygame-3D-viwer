class FileManager:

    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):

        faces, vertices = [], []

        with open(self.filepath) as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(i) for i in line.split()[1:]]+[1])
                elif line.startswith('f'):
                    faces.append([int(p.split('/')[0])-1 for p in line.split()[1:]])

        return faces, vertices
