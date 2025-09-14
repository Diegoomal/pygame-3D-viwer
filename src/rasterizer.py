import numpy as np                                                              # type: ignore
import pygame as pg                                                             # type: ignore


class Rasterizer:

    @staticmethod
    def draw_textured_triangle(surface: pg.Surface, vertex_2d, vertex_3d, face, texture: pg.Surface):

        pts2d = [
            tuple(vertex_2d[i]) 
            for i in face[:3]
        ]
        
        ptsuv = [
            (
                vertex_3d[i][0] % 1, 
                vertex_3d[i][1] % 1
            )
            for i in face[:3]
        ]

        tex_w, tex_h = texture.get_size()
        tex_pixels = pg.surfarray.pixels3d(texture)

        min_x = max(int(min(p[0] for p in pts2d)), 0)
        max_x = min(int(max(p[0] for p in pts2d)), surface.get_width() - 1)
        
        min_y = max(int(min(p[1] for p in pts2d)), 0)
        max_y = min(int(max(p[1] for p in pts2d)), surface.get_height() - 1)

        def edge(p1, p2, p):
            return (p[0] - p1[0]) * (p2[1] - p1[1]) - (p[1] - p1[1]) * (p2[0] - p1[0])

        area = edge(pts2d[0], pts2d[1], pts2d[2])
        surf_array = pg.surfarray.pixels3d(surface)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):

                p = (x + 0.5, y + 0.5)
                w0 = edge(pts2d[1], pts2d[2], p)
                w1 = edge(pts2d[2], pts2d[0], p)
                w2 = edge(pts2d[0], pts2d[1], p)
                
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                
                    b0, b1, b2 = w0 / area, w1 / area, w2 / area
                    u = b0 * ptsuv[0][0] + b1 * ptsuv[1][0] + b2 * ptsuv[2][0]
                    v = b0 * ptsuv[0][1] + b1 * ptsuv[1][1] + b2 * ptsuv[2][1]
                    tx = int(u * tex_w) % tex_w
                    ty = int(v * tex_h) % tex_h
                    surf_array[x, y] = tex_pixels[tx, ty]

        del surf_array
        del tex_pixels



# from numba import njit                                                          # type: ignore


# @njit(fastmath=True)
# def rasterize_triangle(framebuffer, width, height, pts2d, ptsuv, tex_pixels):

#     min_x = max(int(np.min(pts2d[:,0])), 0)
#     max_x = min(int(np.max(pts2d[:,0])), width - 1)
#     min_y = max(int(np.min(pts2d[:,1])), 0)
#     max_y = min(int(np.max(pts2d[:,1])), height - 1)

#     def edge(p1, p2, p):
#         return (p[0]-p1[0])*(p2[1]-p1[1]) - (p[1]-p1[1])*(p2[0]-p1[0])

#     area = edge(pts2d[0], pts2d[1], pts2d[2])
#     tex_h, tex_w, _ = tex_pixels.shape

#     for y in range(min_y, max_y+1):
#         for x in range(min_x, max_x+1):
#             p = (x+0.5, y+0.5)
#             w0 = edge(pts2d[1], pts2d[2], p)
#             w1 = edge(pts2d[2], pts2d[0], p)
#             w2 = edge(pts2d[0], pts2d[1], p)
#             if w0>=0 and w1>=0 and w2>=0:
#                 b0, b1, b2 = w0/area, w1/area, w2/area
#                 u = b0*ptsuv[0][0] + b1*ptsuv[1][0] + b2*ptsuv[2][0]
#                 v = b0*ptsuv[0][1] + b1*ptsuv[1][1] + b2*ptsuv[2][1]
#                 tx = int(u*tex_w) % tex_w
#                 ty = int(v*tex_h) % tex_h
#                 framebuffer[y, x] = tex_pixels[ty, tx]

# class Rasterizer:

#     @staticmethod
#     def draw_textured_triangle(surface: pg.Surface, vertex_2d, vertex_3d, face, texture: pg.Surface):
#         pts2d = np.array([vertex_2d[i] for i in face[:3]], dtype=np.float32)
#         ptsuv = np.array([(vertex_3d[i][0] % 1, vertex_3d[i][1] % 1) for i in face[:3]], dtype=np.float32)

#         tex_pixels = np.array(pg.surfarray.pixels3d(texture)).astype(np.uint8)
#         tex_pixels = np.transpose(tex_pixels, (1,0,2))  # (h,w,3)

#         fb = np.array(pg.surfarray.pixels3d(surface)).astype(np.uint8)
#         fb = np.transpose(fb, (1,0,2))  # (h,w,3)

#         rasterize_triangle(fb, fb.shape[1], fb.shape[0], pts2d, ptsuv, tex_pixels)

#         fb = np.transpose(fb, (1,0,2))
#         pg.surfarray.blit_array(surface, fb)