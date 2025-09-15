import numpy as np                                                              # type: ignore


class LambertShader:

    def __init__(self, light_dir=np.array([0, 0, -1])):
        self.light_dir = light_dir / np.linalg.norm(light_dir)

    def shade(self, face, vertex_3d):
        
        vertex_3d = vertex_3d[:, :3]                                            # x,y,z,w -> drop w component

        vertex_3d_pts = np.array([vertex_3d[i] for i in face])
        
        v1 = vertex_3d_pts[1] - vertex_3d_pts[0]
        v2 = vertex_3d_pts[2] - vertex_3d_pts[0]
        
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        
        intensity = max(0, np.dot(normal, self.light_dir))
        
        val = int(255 * intensity)

        color = (val, val, val)                                                   # R, G, B

        return color
