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

def back_face_culling_v1(vertices, faces, camera_position):

    visible_faces = []

    camera_position = np.array(camera_position[3][:3])                          # new
    
    for face in faces:
        v1, v2, v3 = [np.array(vertices[i][:3]) for i in face[:3]]

        # Calculate the face normal
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal)    # Normalize the normal vector

        # Calculate the vector from the camera to the face
        to_camera = camera_position[:3] - v1

        # Check if the face is visible to the camera
        if np.dot(normal, to_camera) < 0:           # Dot product
            visible_faces.append(face)

    return visible_faces

def back_face_culling_v2(vertices, faces, camera_position):

    visible_faces = []

    camera_position = np.array(camera_position[3][:3])                          # new

    for face in faces:
        v1, v2, v3 = [np.array(vertices[i][:3]) for i in face[:3]]

        # Calculate the face normal
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

        # Calculate the vector from the camera to the face
        to_camera = camera_position[:3] - v1

        # Check if the face is visible to the camera
        if np.dot(normal, to_camera) < 0:  # Dot product
            # Compute the average depth (Z) of the face
            depth = (v1[2] + v2[2] + v3[2]) / 3
            visible_faces.append((depth, face))

    # Sort faces by depth (from farthest to nearest)
    visible_faces.sort(key=lambda x: x[0], reverse=True)

    # Return only the sorted faces
    return [face for _, face in visible_faces]

def back_face_culling_v3(vertices, faces, camera_position):
    """
    Otimiza o processo de Back-Face Culling, removendo polígonos invisíveis
    e ordenando os visíveis por profundidade para uma renderização precisa.

    Args:
        vertices: Lista de vértices do modelo [(x, y, z, ...), ...].
        faces: Lista de faces, onde cada face é uma lista de índices para os vértices [(i1, i2, i3), ...].
        camera_position: Posição da câmera no espaço 3D [x, y, z].

    Returns:
        Lista de faces visíveis ordenadas pela profundidade (de longe para perto).
    """
    visible_faces = []

    camera_position = np.array(camera_position[3][:3])                          # new

    for face in faces:
        # Extração dos vértices da face
        v1, v2, v3 = [np.array(vertices[i][:3]) for i in face[:3]]

        # Calcula o vetor normal da face (produto vetorial)
        normal = np.cross(v2 - v1, v3 - v1)
        normal_length = np.linalg.norm(normal)

        # Evita erros caso a face seja degenerada (normal = 0)
        if normal_length == 0:
            continue
        normal /= normal_length  # Normaliza o vetor normal

        # Vetor da face para a câmera
        to_camera = camera_position[:3] - v1

        # Determina se a face está visível (culling)
        if np.dot(normal, to_camera) < 0:  # Face está voltada para a câmera
            # Calcula a profundidade média (Z) da face
            avg_depth = np.mean([v1[2], v2[2], v3[2]])
            visible_faces.append((avg_depth, face))

    # Ordena as faces visíveis por profundidade (de longe para perto)
    visible_faces.sort(key=lambda x: x[0], reverse=True)

    # Retorna apenas as faces ordenadas
    return [face for _, face in visible_faces]

def back_face_culling_v4(vertices, faces, camera_matrix):

    culled_faces = []
    
    # Extrai a posição da câmera da última linha da matriz de câmera
    camera_position = np.array(camera_matrix[3][:3])

    for face in faces:

        # Garante que apenas triângulos sejam processados
        if len(face) < 3: continue

        # Obtenha os vértices do triângulo
        v1 = np.array(vertices[face[0]])
        v2 = np.array(vertices[face[1]])
        v3 = np.array(vertices[face[2]])

        # Vetores das arestas do triângulo
        edge1 = v2[:3] - v1[:3]
        edge2 = v3[:3] - v1[:3]

        # Normal do triângulo (produto vetorial das arestas)
        normal = np.cross(edge1, edge2)

        # Vetor da câmera para o vértice do triângulo
        view_vector = v1[:3] - camera_position

        # Culling: checar se o triângulo está de frente ou de costas para a câmera
        if np.dot(normal, view_vector) < 0: culled_faces.append(face)

    return culled_faces

def back_face_culling_v5(vertices, faces, camera_matrix):

    culled_faces = []

    camera_position = np.array(camera_matrix[3][:3])

    for face in faces:

        if len(face) < 3: continue

        v1, v2, v3 = [np.array(vertices[i][:3]) for i in face[:3]]

        normal = np.cross(v2 - v1, v3 - v1)
        if normal == 0: continue
        normal = normal / np.linalg.norm(normal)

        to_camera = camera_position[:3] - v1
        
        if np.dot(normal, to_camera) < 0:
            avg_depth = np.mean([v1[2], v2[2], v3[2]])
            culled_faces.append((avg_depth, face))

    culled_faces.sort(key=lambda x: x[0], reverse=True)

    return [face for _, face in culled_faces]

def back_face_culling_v6(vertices, faces, camera_matrix):

    culled_faces = []

    camera_position = np.array(camera_matrix[3][:3])

    for face in faces:
        
        if len(face) < 3: continue

        v1, v2, v3 = [np.array(vertices[i][:3]) for i in face[:3]]

        normal = np.cross(v2 - v1, v3 - v1)
        if np.any(normal == 0): continue
        normal = normal / np.linalg.norm(normal)

        to_camera = camera_position[:3] - v1

        if np.dot(normal, to_camera) < 0:
            avg_depth = np.mean([v1[2], v2[2], v3[2]])
            culled_faces.append((avg_depth, face))

    culled_faces.sort(key=lambda x: x[0], reverse=True)

    return [face for _, face in culled_faces]

def hidden_faces_removal(vertices_3d, faces, matrix_camera):
    """
    Remove faces ocultas usando Z-buffer.
    
    :param vertices_3d: Lista de vértices no espaço 3D (Nx3 array)
    :param faces: Lista de faces representadas por índices dos vértices
    :param matrix_camera: Matriz de projeção da câmera (4x4)
    :return: Lista de faces visíveis
    """
    # Projeta os vértices 3D para o espaço da câmera
    vertices_homog = np.c_[vertices_3d, np.ones(len(vertices_3d))].T
    # vertices_projected = matrix_camera @ vertices_homog
    vertices_projected = matrix_multiply(matrix_camera, vertices_homog)
    vertices_projected /= vertices_projected[3]  # Normalização perspectiva
    

    # Calcula normais das faces
    faces_visible = []
    for face in faces:
        v0, v1, v2 = [vertices_projected[:3, i] for i in face]
        
        # Vetores das arestas
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Normal da face (produto vetorial)
        normal = np.cross(edge1, edge2)
        
        # Verifica se a face está voltada para a câmera
        view_vector = v0  # No espaço da câmera, a câmera está na origem
        if np.dot(normal, view_vector) < 0:
            faces_visible.append(face)
    
    return faces_visible
