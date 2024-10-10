import numpy as np
from typing import Union, Sequence, Tuple, Optional
from my_package.core.geometry import is_rect_in_max_range, find_sectors_indices

def lidar(pose: Union[Sequence, np.ndarray], obstacle: Sequence, lidar_params: dict) -> np.ndarray:
    """
    Simula un sensore Lidar che rileva le distanze dagli ostacoli circostanti.
    Args:
        pose (Union[Sequence, np.ndarray]): La posizione e l'orientamento del sensore Lidar, rappresentati come una sequenza o un array numpy di 3 componenti [x, y, theta].
        obstacle (Sequence):Un ostacolo rappresentato dalle sue 4 componenti (xmin, ymin, xmax, ymax).
        lidar_params (dict): Un dizionario contenente i parametri del sensore Lidar con le chiavi 'max_range' (distanza massima), 'n_beams' (numero di raggi), e 'FoV' (campo visivo).
    Returns:
        np.ndarray: Un array numpy contenente le distanze rilevate per ciascun raggio del sensore Lidar.
    Raises:
        ValueError: Se 'pose' non è una sequenza di 3 componenti.
        ValueError: Se 'obstacles' non è composto 4 componenti.
        ValueError: Se 'lidar_params' non è un dizionario con le chiavi 'max_range', 'n_beams', 'FoV'.
    """
    # Check 1: pose must be a list of 3 components
    if not isinstance(pose, np.ndarray) or len(pose) != 3:
        raise ValueError("pose must be a list of 3 components")

    # Check 2: obstacles must be of type (xmin, ymin, xmax, ymax)
    if not isinstance(obstacle, Sequence) or len(obstacle) != 4:
        raise ValueError("obstacles must be a Sequence of type (xmin, ymin, xmax, ymax)")

    # Check 3: lidar_params must be a dictionary with keys 'max_range', 'n_beams', 'FoV'
    if not isinstance(lidar_params, dict) or not all(key in lidar_params for key in ['max_range', 'n_beams', 'FoV']):
        raise ValueError("lidar_params must be a dictionary with keys 'max_range', 'n_beams', 'FoV'")
    
    # Estrae i parametri del sensore Lidar
    x, y, theta = pose
    max_range = lidar_params['max_range']
    n_beams = lidar_params['n_beams']
    FoV = lidar_params['FoV']

    # Inizializza le distanze al valore massimo
    ranges = np.full(n_beams, max_range)

    # Angoli assoluti dei raggi
    angles = np.linspace(- FoV / 2, FoV / 2, n_beams) + theta

    for i, angle in enumerate(angles):
        min_distance = max_range
        distance = ray_rectangle_intersection(x, y, angle, obstacle)
        if distance is not None and distance < min_distance:
            min_distance = distance
        ranges[i] = min_distance
    return ranges

def ray_rectangle_intersection(x0: float, y0: float, angle: float, obstacle: Sequence) -> Union[float, None]:
    """
    Calcola la distanza la distanza da percorrere lungo il raggio per intersecare un ostacolo rettangolare.
    Parametri:
    - x0 (float): La coordinata x dell'origine del raggio.
    - y0 (float): La coordinata y dell'origine del raggio.
    - angle (float): L'angolo del raggio in radianti.
    - obstacle (tuple): Una tupla contenente le coordinate dell'ostacolo rettangolare nel formato (x_min, y_min, x_max, y_max).
    Ritorna:
    - min_distance (float): La distanza minima tra il raggio e l'ostacolo rettangolare.
    Nota:
    - Il raggio è definito dalla sua origine (x0, y0) e dal suo angolo.
    - L'ostacolo è definito dalle sue coordinate minime e massime x e y.
    - La funzione calcola il punto di intersezione tra il raggio e ciascun lato dell'ostacolo rettangolare.
    - La distanza minima è la distanza più breve tra l'origine del raggio e qualsiasi punto di intersezione.
    """
    # Estrae le coordinate dell'ostacolo
    x_min, y_min, x_max, y_max = obstacle
    # Calcola i 4 lati dell'ostacolo (segmenti)
    edges = [((x_min, y_min), (x_max, y_min)),
                ((x_max, y_min), (x_max, y_max)),
                ((x_max, y_max), (x_min, y_max)),
                ((x_min, y_max), (x_min, y_min))]
    # Inizializza la distanza minima a None
    min_distance = None
    # Direzione del raggio
    dx = np.cos(angle)
    dy = np.sin(angle)
    # Itera sui lati dell'ostacolo
    for edge in edges:
        # Calcola il punto di intersezione tra il raggio e il lato
        point = ray_segment_intersection(x0, y0, dx, dy, edge)
        if point is not None:
            # Calcola la distanza tra il punto di intersezione e l'origine
            distance = np.hypot(point[0] - x0, point[1] - y0)
            if min_distance is None or distance < min_distance:
                min_distance = distance
    return min_distance

def ray_segment_intersection(x0: float, y0: float, dx: float, dy: float, segment: Tuple) -> Union[Tuple, None]:
    """
    Calcola il punto di intersezione tra un raggio e un segmento di linea.
    Parametri:
    - x0 (float): Coordinata x del punto di partenza del raggio.
    - y0 (float): Coordinata y del punto di partenza del raggio.
    - dx (float): Componente x del vettore direzione del raggio.
    - dy (float): Componente y del vettore direzione del raggio.
    - segment (tuple): Tupla contenente le coordinate degli estremi del segmento di linea nel formato ((x1, y1), (x2, y2)).
    Ritorna:
    - tuple o None: Se esiste un punto di intersezione, restituisce una tupla (ix, iy) contenente le coordinate x e y del punto di intersezione. Se non esiste un punto di intersezione, restituisce None.
    """
    # Estrae le coordinate del segmento
    (x1, y1), (x2, y2) = segment
    x3, y3 = x0, y0
    x4, y4 = x0 + dx, y0 + dy
    
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(den) <= 1e-9:
        return None # Le linee sono parallele o sovrapposte
    
    # Calcola i parametri di intersezione
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    if 0 <= t <= 1 and u >= 0:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    else:
        return None

def discretize_lidar(pose: np.ndarray, lidar_params: dict, obstacle: Sequence, safe_distance: float, sector_indices: np.ndarray) -> Tuple:
    """
    Discretizza i dati del LiDAR in settori e verifica la presenza di ostacoli e collisioni.
    Args:
        pose (np.ndarray): La posizione attuale del LiDAR.
        lidar_params (dict): Parametri del LiDAR, inclusi 'max_range' e 'FoV'.
        obstacle (Sequence): Lista degli ostacoli presenti nell'ambiente.
        safe_distance (float): Distanza di sicurezza per rilevare una possibile collisione.
    Returns:
        Tuple: Una tupla contenente un array di settori occupati e un flag di allerta collisione.
    """
    # Init output 
    n_sectors = len(sector_indices)
    collision_alert = 0
    occupied_sectors = np.zeros(n_sectors)

    # Se il rettagolo non è nel range massimo del LiDAR, return immediato
    max_range = lidar_params['max_range']
    if not is_rect_in_max_range(obstacle, pose, max_range):
        return occupied_sectors, collision_alert

    # Superato il check simulo il LiDAR
    ranges = lidar(pose, obstacle, lidar_params)
    sectors = np.zeros(n_sectors, dtype=object)

    for i in range(n_sectors):
        sectors[i] = ranges[sector_indices[i][0]:sector_indices[i][1]]
        if np.min(sectors[i]) < max_range:
            occupied_sectors[i] = 1

    if np.min(ranges) <= safe_distance:
        collision_alert = 1

    return occupied_sectors, collision_alert

def proximity_sensor(pose: np.ndarray, heading: float, max_range: float, obstacle: Sequence, return_type: Optional[str] = 'binary') -> int | float:
    """
    Simula un sensore di prossimità binario che rileva ostacoli entro una distanza di sicurezza.
    Args:
        pose (np.ndarray): La posizione attuale del sensore come array numpy [x, y, theta].
        heading (float): L'angolo di orientamento del sensore rispetto alla posizione attuale.
        max_range (float): La distanza di sicurezza entro la quale un ostacolo viene rilevato.
        obstacle (Sequence): La rappresentazione dell'ostacolo come una sequenza di coordinate.
        return_type (Optional[str]): Il tipo di valore restituito dal sensore di prossimità. Può essere 'binary' (default) o 'range'.
    Returns:
        int | float: Un valore binario o una distanza, a seconda del tipo di valore restituito richiesto.
    """
    valid_return_types = ['binary', 'range']
    if return_type not in valid_return_types:
        raise ValueError(f"return_type must be one of {valid_return_types}")
    theta = pose[2] + heading
    distance = ray_rectangle_intersection(pose[0], pose[1], theta, obstacle)


    if distance is not None and distance <= max_range:

        match(return_type):
            case 'range':
                return distance
            case 'binary':
                return 1
            case _:
                return 1
            
    else:

        match(return_type):
            case 'range':
                return max_range
            case 'binary':
                return 0
            case _:
                return 0

if __name__ == "__main__":
    # Define the pose of the lidar sensor
    pose = np.array([1.5, 0.51, np.pi])

    # Define the obstacles in the environment
    obstacle = (1,1,2,2)

    # Define the lidar parameters
    lidar_params = {
        'max_range': 1.6,
        'n_beams': 10,
        'FoV': np.deg2rad(60)
    }
    sector_indices = find_sectors_indices(lidar_params['n_beams'], 3)

    # discard_obstacle(pose, 1.0, obstacles)
    # Call the lidar function to get the range measurements
    ranges = lidar(pose, obstacle, lidar_params)
    sectors, alert = discretize_lidar(pose, lidar_params, obstacle, 0.5, sector_indices)

    # Print the range measurements
    print("Range measurements:", ranges)

    alert = proximity_sensor(pose, -np.pi/2, 0.5, obstacle)
    print("Proximity alert:", alert)