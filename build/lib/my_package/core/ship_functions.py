import numpy as np
from scipy.integrate import quad
from typing import Optional, Dict
import matplotlib.pyplot as plt

class ShipSegment:
    def __init__(self, id: int, start_point: tuple, end_point: tuple):
        self.start_point = start_point
        self.end_point = end_point
        self.mid_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
        self.id = id
        self.seen = False


class ShipObstacle:
    def __init__(self, Options: Dict = {}, use_default_values: bool = True):
        if use_default_values:
            L =                1
            W =                1.5
            a_left =           0.2
            a_right =          2.8
            desired_distance = 0.2
        else:
            L =                Options['L']
            W =                Options['W']
            a_left =           Options['a_left']
            a_right =          Options['a_right']
            desired_distance = Options['desired_distance']
        
        self.default_points = generate_ship_polygon(L, W, a_left, a_right, desired_distance)
        self.points = self.default_points

        # Estremi del poligono
        self.x_max = max([point[0] for point in self.points])
        self.x_min = min([point[0] for point in self.points])
        self.y_max = max([point[1] for point in self.points])
        self.y_min = min([point[1] for point in self.points])

        # Centro e raggio del poligono
        Cx = (self.x_max + self.x_min) / 2
        Cy = (self.y_max + self.y_min) / 2
        self.center = (Cx, Cy)
        self.radius = np.sqrt((self.x_max - Cx)**2 + (self.y_max - Cy)**2)

        # Genera i segmenti
        self.generate_segments()

    def generate_segments(self):
        self.segments_dict = {}
        for i in range(len(self.points)):
            curr_point = self.points[i]
            if i == len(self.points) - 1:
                next_point = self.points[0]
            else:
                next_point = self.points[i + 1]
            segment = ShipSegment(i, curr_point, next_point)
            self.segments_dict[i] = segment

    def reset_ship(self):
        for segment in self.segments_dict.values():
            segment.seen = False
    
    def rotate_ship(self, angle: float) -> None:
        # Calcolo nuovi punti ruotati
        rot_points = []
        for point in self.points:
            point = rotate_point(point, self.center, angle)
            rot_points.append(point)
        self.points = rot_points

        self.x_max = max([point[0] for point in self.points])
        self.x_min = min([point[0] for point in self.points])
        self.y_max = max([point[1] for point in self.points])
        self.y_min = min([point[1] for point in self.points])

        # Calcolo nuovi segmenti ruotati
        for segment in self.segments_dict.values():
            segment.start_point = rotate_point(segment.start_point, self.center, angle)
            segment.end_point = rotate_point(segment.end_point, self.center, angle)
            segment.mid_point = rotate_point(segment.mid_point, self.center, angle)

    def translate_ship(self, vector: tuple) -> None:
        dx, dy = vector
        translated_points = []
        for point in self.points:
            point = translate_point(point, vector)
            translated_points.append(point)
        self.points = translated_points

        self.center = (self.center[0] + dx, self.center[1] + dy)
        self.x_max += dx
        self.x_min += dx
        self.y_max += dy
        self.y_min += dy

        for segment in self.segments_dict.values():
            segment.start_point = translate_point(segment.start_point, vector)
            segment.end_point = translate_point(segment.end_point, vector)
            segment.mid_point = translate_point(segment.mid_point, vector)

    def rototranslate_ship(self, angle: float, distance: tuple):
        self.rotate_ship(angle)
        self.translate_ship(distance)

def translate_point(point: tuple, r: tuple) -> tuple:
    """
    Trasla un punto di una certa distanza.
    Args:
        point (tuple): Una tupla contenente le coordinate (x, y) del punto da traslare.
        r (tuple): Una tupla contenente le distanze (dx, dy) di traslazione lungo gli assi x e y.
    Returns:
        tuple: Una tupla contenente le nuove coordinate (x, y) del punto traslato.
    """

    x, y = point
    dx, dy = r
    return (x + dx, y + dy)

def rotate_point(point: tuple, center: tuple, angle: float) -> tuple:
    """
    Ruota un punto rispetto a un altro punto.
    Args:
        point (tuple): Le coordinate del punto da ruotare (x, y).
        center (tuple): Le coordinate del punto intorno a cui ruotare (x, y).
        angle (float): L'angolo di rotazione in radianti.
    Returns:
        tuple: Le coordinate del punto ruotato (x, y).
    """
    x, y = point
    cx, cy = center
    x_rot = (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle) + cx
    y_rot = (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle) + cy
    return (x_rot, y_rot)

def arc_length_elliptical(theta: float, a: float, b: float) -> float:
    """
    Calcola la lunghezza dell'arco di un'ellisse data un'angolo theta e i semiassi a e b.
    Args:
        theta (float): L'angolo in radianti.
        a (float): Il semiasse maggiore dell'ellisse.
        b (float): Il semiasse minore dell'ellisse.
    Returns:
        float: La lunghezza dell'arco dell'ellisse per l'angolo dato.
    """
    dx_dtheta = a * np.sin(theta)
    dy_dtheta = b * np.cos(theta)
    return np.sqrt(dx_dtheta**2 + dy_dtheta**2)

def polygon_from_arc(a: float, b: float, center: tuple, theta_start: float, theta_end: float, desired_distance: float, clockwise: bool = True, N_points: int = 50)-> tuple:
    """
    Genera un poligono approssimato da un arco ellittico.
    Args:
        a (float): Semi-asse maggiore dell'ellisse.
        b (float): Semi-asse minore dell'ellisse.
        center (tuple): Coordinate del centro dell'ellisse (x, y).
        theta_start (float): Angolo iniziale dell'arco in radianti.
        theta_end (float): Angolo finale dell'arco in radianti.
        clockwise (bool, opzionale): Direzione dell'arco. True per orario, False per antiorario. Default è True.
        N_points (int, opzionale): Numero di punti per approssimare l'arco. Default è 50.
    Returns:
        tuple: Due array numpy contenenti le coordinate x e y dei punti del poligono.
    """
    theta = np.linspace(theta_start, theta_end, N_points)

    arc_lengths = np.array(
        [quad(arc_length_elliptical, theta_start, theta, args=(a, b))[0] for theta in theta]
    )
    total_arc_length = arc_lengths[-1]
    num_points = int(np.ceil(total_arc_length / desired_distance))

    equi_arc_length = np.linspace(0, total_arc_length, num_points)

    if clockwise:
        equi_theta = np.interp(equi_arc_length, arc_lengths, theta)
    else:
        equi_theta = np.interp(equi_arc_length, arc_lengths, theta)[::-1]

    if clockwise:
        x = center[0] - a * np.cos(equi_theta)
        y = center[1] + b * np.sin(equi_theta)
    else:
        x = center[0] + a * np.cos(equi_theta)
        y = center[1] + b * np.sin(equi_theta)

    return x, y

def generate_ship_polygon(L: float, W: float, a_left: float, a_right: float, desired_distance: float) -> list:
    """
    Genera un poligono che rappresenta una nave basato sui parametri forniti.
    Il poligono è composto da un rettangolo con due semicerchi ai lati e segmenti equidistanti sulla parte superiore e inferiore.
    Args:
        L (float): Lunghezza della parte rettilinea della nave.
        W (float): Larghezza della nave.
        a_left (float): Semiasse x dell'arco sinistro.
        a_right (float): Semiasse x dell'arco destro.
        desired_distance (float): Distanza desiderata tra i punti del poligono.
    Returns:
        list: Una lista di tuple contenenti le coordinate (x, y) dei punti del poligono.
    Raises:
        ValueError: Se il numero di punti x e y non è lo stesso.
    """

    b_left = W / 2
    b_right = W / 2

    center_left = (-L/2, 0)
    center_right = (L/2, 0)

    x_left, y_left = polygon_from_arc(a_left, b_left, center_left, -np.pi/2, np.pi/2, desired_distance=desired_distance, clockwise=True)
    x_right, y_right = polygon_from_arc(a_right, b_right, center_right, -np.pi/2, np.pi/2, desired_distance=desired_distance, clockwise=False)

    num_points_center = int(np.ceil(L / desired_distance))

    x_center_top = np.linspace(-L/2, L/2, num_points_center)
    x_center_bottom = np.linspace(L/2, -L/2, num_points_center)
    y_center_top = np.full_like(x_center_top, W/2)
    y_center_bottom = np.full_like(x_center_bottom, -W/2)

    x = np.concatenate([x_left, x_center_top[1:-1], x_right, x_center_bottom[1:-1]])
    y = np.concatenate([y_left, y_center_top[1:-1], y_right, y_center_bottom[1:-1]])

    points = []
    if len(x) != len(y):
        raise ValueError("The number of x and y points must be the same.")
    else:
    
        for i in range(len(x)):
            x_round = round(float(x[i]), 2)
            y_round = round(float(y[i]), 2)
            points.append((x_round, y_round))
    return points

if __name__ == "__main__":
    """ Test della funzione generate_ship_polygon """
    # L = 1
    # W = 1.5
    # a_left = 0.2
    # a_right = 2.8
    # desired_distance = 0.2
    # points = generate_ship_polygon(L, W, a_left, a_right, desired_distance)
    # print(len(points))
    # x, y = zip(*points)
    # rot_points = []
    # for point in points:
    #     x_rot, y_rot = rotate_point(point, (1, 0), -np.pi/4)
    #     rot_points.append((x_rot, y_rot))

    # x_rot, y_rot = zip(*rot_points)
    # # Grafico della forma aggiornata
    # plt.figure(figsize=(8, 4))
    # plt.plot(x, y, '-o', label='Rettangolo con Semicerchi e Segmenti Equidistanti')
    # plt.plot(x_rot, y_rot, '-o', label='Rettangolo con Semicerchi e Segmenti Equidistanti Ruotato di 45°')
    # plt.axis('equal')
    # plt.title('Approssimazione della Chiglia con Segmenti Equidistanti sui Lati Orizzontali')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid(True)
    # plt.show()

    """" Test della classe ShipObstacle """
    Ship = ShipObstacle()
    Cx, Cy = Ship.center
    points = Ship.points
    x, y = zip(*points)


    Ship.rototranslate_ship(np.pi/4, (1,1))
    rot_points = Ship.points
    x_rot, y_rot = zip(*rot_points)
    
    dict_points = []
    for segment in Ship.segments_dict.values():
        x_c, y_c = segment.mid_point
        dict_points.append((x_c, y_c))
    x_c, y_c = zip(*dict_points)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, '-o', label='Rettangolo con Semicerchi e Segmenti Equidistanti')
    plt.plot(x_rot, y_rot, '-o', label='Rettangolo con Semicerchi e Segmenti Equidistanti Ruotato di 45°')
    plt.scatter(Cx, Cy, color='r', label='Centro della Nave', s=100)
    plt.plot(x_c, y_c, '+r', label='Punti Medi dei Segmenti')
    plt.axis('equal')
    plt.title('Approssimazione della Chiglia con Segmenti Equidistanti sui Lati Orizzontali')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()
