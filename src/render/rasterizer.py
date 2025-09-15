import numpy as np                                                              # type: ignore
import pygame as pg                                                             # type: ignore


class Rasterizer:

    @staticmethod
    def draw_textured_triangle(surface: pg.Surface,
                               vertex_2d,
                               vertex_clip,
                               vertex_ndc,
                               face,
                               texture: pg.Surface,
                               zbuffer=None):

        # screen-space verts
        pts2d = [
            tuple(vertex_2d[i]) 
            for i in face[:3]
        ]

        # UVs derived from clip (u,v in 0..1)
        ptsuv = [
            (
                (vertex_clip[i][0] / vertex_clip[i][3]) % 1.0,
                (vertex_clip[i][1] / vertex_clip[i][3]) % 1.0
            )
            for i in face[:3]
        ]

        # clip-space w and ndc z (depth)
        ptsw = [vertex_clip[i][3] for i in face[:3]]
        ptsz = [vertex_ndc[i][2] for i in face[:3]]

        tex_w, tex_h = texture.get_size()
        tex_pixels = pg.surfarray.pixels3d(texture)
        surf_array = pg.surfarray.pixels3d(surface)

        min_x = max(int(min(p[0] for p in pts2d)), 0)
        max_x = min(int(max(p[0] for p in pts2d)), surface.get_width() - 1)
        min_y = max(int(min(p[1] for p in pts2d)), 0)
        max_y = min(int(max(p[1] for p in pts2d)), surface.get_height() - 1)

        def edge(p1, p2, p):
            return (p[0] - p1[0]) * (p2[1] - p1[1]) - (p[1] - p1[1]) * (p2[0] - p1[0])

        area = edge(pts2d[0], pts2d[1], pts2d[2])
        if abs(area) < 1e-9: del surf_array; del tex_pixels; return;

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):

                p = (x + 0.5, y + 0.5)
                w0 = edge(pts2d[1], pts2d[2], p)
                w1 = edge(pts2d[2], pts2d[0], p)
                w2 = edge(pts2d[0], pts2d[1], p)

                inside = (w0 >= 0 and w1 >= 0 and w2 >= 0) if area > 0 else (w0 <= 0 and w1 <= 0 and w2 <= 0)
                if not inside:
                    continue

                b0, b1, b2 = w0 / area, w1 / area, w2 / area

                # perspective-correct interpolation of (u,v)
                inv_w = b0 / ptsw[0] + b1 / ptsw[1] + b2 / ptsw[2]
                if inv_w == 0:
                    continue

                u = (b0 * ptsuv[0][0] / ptsw[0] + b1 * ptsuv[1][0] / ptsw[1] + b2 * ptsuv[2][0] / ptsw[2]) / inv_w
                v = (b0 * ptsuv[0][1] / ptsw[0] + b1 * ptsuv[1][1] / ptsw[1] + b2 * ptsuv[2][1] / ptsw[2]) / inv_w

                # depth from NDC z (lower typically means nearer)
                depth = b0 * ptsz[0] + b1 * ptsz[1] + b2 * ptsz[2]

                if zbuffer is None or depth < zbuffer[x, y]:
                    if zbuffer is not None:
                        zbuffer[x, y] = depth
                    tx = int(u * tex_w) % tex_w
                    ty = int(v * tex_h) % tex_h
                    surf_array[x, y] = tex_pixels[tx, ty]

        del surf_array
        del tex_pixels
