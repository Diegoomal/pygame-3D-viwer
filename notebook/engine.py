import math
import numpy as np                  # type: ignore
import pygame as pg                 # type: ignore
from PIL import Image, ImageDraw    # type: ignore

WIDTH, HEIGHT = 1600, 900

def is_out_of_bounds(arr, width=WIDTH, height=HEIGHT):
    return any((x < 0 or x > width or y < 0 or y > height) for x, y in arr)

def get_rotation_matrix_by_axis(angle=0.0, axis=0):
    s, c = math.sin(angle), math.cos(angle)
    if axis == 0: 
        matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]]  # eixo X (pitch)
    elif axis == 1: 
        matrix = [[c, 0.0, -s, 0.0], [0.0, 1.0, 0.0, 0.0], [s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]]  # eixo Y (yaw)
    elif axis == 2: 
        matrix = [[c, s, 0.0, 0.0], [-s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]  # eixo Z (roll)
    else: 
        matrix = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    return matrix

def get_rotation_matrix_axis_x(angle=0.0):
    s, c = math.sin(angle), math.cos(angle)
    matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
    return matrix

def get_rotation_matrix_axis_y(angle=0.0):
    s, c = math.sin(angle), math.cos(angle)
    matrix = [[c, 0.0, -s, 0.0], [0.0, 1.0, 0.0, 0.0], [s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
    return matrix

def get_rotation_matrix_axis_z(angle=0.0):
    s, c = math.sin(angle), math.cos(angle)
    matrix = [[c, s, 0.0, 0.0], [-s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    return matrix

def get_translation_matrix(position):
    x, y, z = position
    matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [x, y, z, 1]]
    return matrix

def get_matrix_camera(start_position=(0, 0, -5)):
    x, y, z = start_position
    matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [x, y, z, 1]]
    return matrix

def get_screen_matrix(width=WIDTH, height=HEIGHT):
    hw, hh = width // 2, height // 2
    screen_matrix = [[hw, 0, 0, 0], [0, -hh, 0, 0], [0, 0, 1, 0], [hw, hh, 0, 1]]
    return screen_matrix

def get_matrix_projection(width=WIDTH, height=HEIGHT, near_plane=0.1, far_plane=100):
    h_fov = math.pi / 3
    v_fov = h_fov * (height / width)

    m00 = 1 / math.tan(h_fov / 2)
    m11 = 1 / math.tan(v_fov / 2)

    m22 = -(far_plane + near_plane) / (far_plane - near_plane)
    m32 = -2 * near_plane * far_plane / (far_plane - near_plane)

    projection_matrix = [[m00, 0, 0, 0], [0, m11, 0, 0], [0, 0, m22, -1], [0, 0, m32, 0]]
    return projection_matrix

def process_polygon_3d(vertices, camera_matrix, angle):
    
    # m_vertices = (
    #     vertices
    #     @ get_rotation_matrix_axis_x(angle)
    #     @ get_rotation_matrix_axis_y(angle)
    #     @ get_rotation_matrix_axis_z(angle)
    #     @ camera_matrix
    # )

    # return m_vertices

    mv = vertices
    mrx = get_rotation_matrix_axis_x(angle)
    mry = get_rotation_matrix_axis_y(angle)
    mrz = get_rotation_matrix_axis_z(angle)
    mc = camera_matrix

    m_res = matrix_multiply(mv, mrx)
    m_res = matrix_multiply(m_res, mry)
    m_res = matrix_multiply(m_res, mrz)
    m_res = matrix_multiply(m_res, mc)

    return m_res

def process_polygon_2d(m_vertices_3d, m_projection, m_screen):

    t_vertices = matrix_multiply(m_vertices_3d, m_projection)

    # Divisão pela coordenada homogênea (perspectiva)
    for vertex in t_vertices:
        vertex[:] = [coord / vertex[-1] for coord in vertex]

    t_vertices = matrix_multiply(t_vertices, m_screen)
    transformed_vertices_2d = [vertex[:2] for vertex in t_vertices]
    return transformed_vertices_2d

def check_valid_poligons(faces, vertices, width=WIDTH, height=HEIGHT):
    valid_polygons = []

    for f in faces:
        p = [vertices[i] for i in f]

        if not is_out_of_bounds(p, width, height):
            valid_polygons.append(p)

    return valid_polygons

def matrix_multiply(a, b):
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    result = [[0] * cols_b for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            result[i][j] = sum(a[i][k] * b[k][j] for k in range(cols_a))

    return result

def create_image(polygons):

    # polygon
    polygons = np.array(polygons, dtype=np.float32)

    # Configuração da imagem
    image = Image.new("RGB", (1600, 900), "black")  # Fundo preto
    draw = ImageDraw.Draw(image)
    
    # Desenha os polígonos na imagem
    [
        draw.polygon(p, outline="orange")
        for p in polygons
        if len(polygons) > 0
    ]

    return image

def get_model_2d():

    model = {
        "vertices": [ 
            # x, y, z, w
            ( 0, 0, 0, 1 ),  # v0
            ( 1, 0, 0, 1 ),  # v1
            ( 1, 1, 0, 1 ),  # v2
            ( 0, 1, 0, 1 )   # v3
        ],
        "faces": [
            [0, 1, 2],      # Triângulo 1
            [0, 2, 3]       # Triângulo 2
        ]
    }

    return model['faces'], model['vertices'] 

def get_model_3d():

    model = {
        "vertices": [
            (0, 0, 0, 1),  # v0
            (1, 0, 0, 1),  # v1
            (1, 1, 0, 1),  # v2
            (0, 1, 0, 1),  # v3
            (0, 0, 1, 1),  # v4
            (1, 0, 1, 1),  # v5
            (1, 1, 1, 1),  # v6
            (0, 1, 1, 1)   # v7
        ],
        "faces": [
            [0, 1, 2], [0, 2, 3],   # Face inferior
            [4, 5, 6], [4, 6, 7],   # Face superior
            [0, 1, 5], [0, 5, 4],   # Face frontal
            [1, 2, 6], [1, 6, 5],   # Face direita
            [2, 3, 7], [2, 7, 6],   # Face traseira
            [3, 0, 4], [3, 4, 7]    # Face esquerda
        ]
    }

    return model['faces'], model['vertices']

def read_model(filename:str = 'suzanne'):

    faces: list[list[int]] = []
    vertices: list[list[float]] = []
    filepath:str = f'../models/{filename}/model.obj'
    
    with open(filepath) as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(i) for i in line.split()[1:]] + [1])
            elif line.startswith('f'):
                faces.append([int(f.split('/')[0]) - 1 for f in line.split()[1:]])

    return faces, vertices
