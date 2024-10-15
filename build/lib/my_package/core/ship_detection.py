import numpy as np
from typing import Union, Sequence, Tuple
import matplotlib.pyplot as plt

from my_package.core import ShipObstacle
from my_package.core import is_rect_in_max_range
from my_package.core import ray_segment_intersection

def lidar(pose: np.ndarray, Ship: ShipObstacle, lidar_params: dict) -> Tuple[np.ndarray, np.ndarray, set[int]]:
    """
    Simula un sensore LiDAR per rilevare un ostacolo definito da un oggetto ShipObstacle.
    Args:
        pose (np.ndarray): La posa del sensore LiDAR, rappresentata come un array numpy [x, y, theta].
        Ship (ShipObstacle): L'ostacolo da rilevare.
        lidar_params (dict): Parametri del LiDAR, inclusi 'max_range' (portata massima), 
                             'n_beams' (numero di raggi) e 'FoV' (campo visivo).
    Returns:
        Tuple[np.ndarray, np.ndarray, set[int]]: Una tupla contenente:
            - ranges (np.ndarray): Distanze rilevate per ciascun raggio.
            - angles (np.ndarray): Angoli dei raggi rispetto alla posa del sensore.
            - seen_segments (set[int]): Insieme degli ID dei segmenti visti.
    """

    # Import dati lidar
    max_range = lidar_params['max_range']
    n_beams = lidar_params['n_beams']
    FoV = lidar_params['FoV']

    # Init dei ranges al valore massimo
    ranges = np.full(n_beams, max_range, dtype=np.float64)
    angles = np.linspace(- FoV / 2, FoV / 2, n_beams) + pose[2]
    seen_segments: set[int] = set()

    # Se la circonferenza non è in max_range, ritorna i ranges iniziali
    ship_center = np.array([Ship.center[0], Ship.center[1]])
    if np.linalg.norm(pose[:2] - ship_center) > (max_range + Ship.radius):
        return ranges, angles, seen_segments

    # Se la bounding box non è in max_range, ritorna i ranges iniziali
    ship_rect = Ship.get_bounding_box()
    if not is_rect_in_max_range(ship_rect, pose, max_range):
        return ranges, angles, seen_segments
    

    # Angoli assoluti dei raggi
    angles = np.linspace(- FoV / 2, FoV / 2, n_beams) + pose[2]
    for i, angle in enumerate(angles):
        min_distance = max_range
        distance, seen_segment_id = ray_polygon_intersection(pose, angle, Ship)
        if distance is not None and distance < min_distance:
            assert seen_segment_id is not None
            seen_segments.add(seen_segment_id)
            min_distance = distance
        ranges[i] = min_distance

    return ranges, angles, seen_segments

def proximity_sensor(pose: np.ndarray, heading: float, max_range: float, Ship: ShipObstacle) -> float:
    """
    Calcola la distanza tra una nave e un ostacolo utilizzando un sensore di prossimità.
    Args:
        pose (np.ndarray): La posizione attuale dell'agente [x, y, theta].
        heading (float): Heading del sensore in radianti.
        max_range (float): La distanza massima che il sensore può rilevare.
        Ship (ShipObstacle): Ostacolo da rilevare.
    Returns:
        float: La distanza rilevata tra la nave e l'ostacolo. Se non ci sono ostacoli entro il raggio massimo, restituisce max_range.
    """
    angle = heading + pose[2]
    min_distance = max_range
    distance, _ = ray_polygon_intersection(pose, angle, Ship)
    if distance is not None and distance < min_distance:
        min_distance = distance

    return min_distance

def ray_polygon_intersection(pose: np.ndarray, angle: float, Ship: ShipObstacle) -> Tuple[Union[float, None], Union[int, None]]:
    """
    Calcola l'intersezione tra un raggio e un poligono rappresentato da un oggetto ShipObstacle.
    Args:
        pose (np.ndarray): La posizione iniziale del raggio come array numpy [x, y].
        angle (float): L'angolo del raggio in radianti.
        Ship (ShipObstacle): L'oggetto ShipObstacle che contiene i segmenti del poligono.
    Returns:
        Tuple[Union[float, None], Union[int, None]]: Una tupla contenente la distanza minima dall'origine del raggio al punto 
        di intersezione più vicino (o None se non ci sono intersezioni) e l'ID del segmento visto (o None se non ci sono intersezioni).
    """
    x0, y0 = pose[:2]
    min_distance = None
    dx = np.cos(angle)
    dy = np.sin(angle)

    seen_segment_id = None

    for segment in Ship.segments_dict.values():
        edge = (segment.start_point, segment.end_point)
        point = ray_segment_intersection(x0, y0, dx, dy, edge)
        if point is not None:
            distance = np.hypot(point[0] - x0, point[1] - y0)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                seen_segment_id = segment.id

    return min_distance, seen_segment_id

if __name__ == "__main__":
    Ship = ShipObstacle((3,0))
    Cx, Cy = Ship.center
    radius = Ship.radius
    points = Ship.points
    x_ship, y_ship = zip(*points)

    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = Cx + radius * np.cos(theta)
    y_circle = Cy + radius * np.sin(theta)

    x_min = Ship.x_min
    x_max = Ship.x_max
    y_min = Ship.y_min
    y_max = Ship.y_max
    square_points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]
    x_square, y_square = zip(*square_points)

    pose = np.array([0, 0, -np.pi/2])
    lidar_params = {'max_range': 2, 'n_beams': 20, 'FoV': np.pi/2}
    FoV = lidar_params['FoV']
    theta_lidar = np.linspace(- FoV / 2, FoV / 2, 100) + pose[2]

    x_range = pose[0] + lidar_params['max_range'] * np.cos(theta_lidar)
    y_range = pose[1] + lidar_params['max_range'] * np.sin(theta_lidar)

    ranges, angles, seen_segments_id = lidar(pose, Ship, lidar_params)
    x_lidar = pose[0] + ranges * np.cos(angles)
    y_lidar = pose[1] + ranges * np.sin(angles)
    print(ranges)
    print(seen_segments_id)

    proximity_sensor_heading = np.pi/2
    proximity_sensor_max_range = 2
    distance = proximity_sensor(pose, proximity_sensor_heading, proximity_sensor_max_range, Ship)
    x_proximity = pose[0] + distance * np.cos(proximity_sensor_heading + pose[2])
    y_proximity = pose[1] + distance * np.sin(proximity_sensor_heading+ pose[2])
    print(distance)

    segment_points = []
    if len(seen_segments_id) > 0:
        for id in seen_segments_id:
            segment = Ship.segments_dict[id]
            x_s, y_s = segment.mid_point
            segment_points.append((x_s, y_s))
        x_segment, y_segment = zip(*segment_points)

    plt.figure(figsize=(8, 4))
    plt.scatter(pose[0], pose[1], color='b', label='Posizione Lidar')
    plt.plot(x_range, y_range, '--')
    plt.scatter(x_lidar, y_lidar, color='g', label='Misurazioni Lidar')
    plt.plot(x_ship, y_ship, label='Nave')
    if len(seen_segments_id) > 0:
        plt.plot(x_segment, y_segment, '+r', label='Segmento visto')
    # plt.scatter(Cx, Cy, color='r', label='Centro della Nave', s=100)
    # plt.plot(x_circle, y_circle, '--', label='Circonferenza di Bounding Box')
    # plt.plot(x_square, y_square, '--', label='Bounding Box')
    plt.scatter(x_proximity, y_proximity, color='r', label='Misurazione Prossimità')
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()
