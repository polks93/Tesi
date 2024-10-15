import numpy as np
import copy
import matplotlib.pyplot as plt

from scipy.integrate import quad
from typing import Optional, Dict

class ShipSegment:
    """
    Rappresenta un segmento del primetro di una nave con un punto di inizio, 
    un punto di fine e un punto medio.
    Attributi:
    ----------
    id : int
        Identificatore univoco del segmento.
    start_point : tuple
        Coordinata del punto di inizio del segmento (x, y).
    end_point : tuple
        Coordinata del punto di fine del segmento (x, y).
    mid_point : tuple
        Coordinata del punto medio del segmento (x, y).
    seen : bool
        Indica se il segmento è stato visto o meno. Inizialmente impostato a False.
    """
    def __init__(self, id: int, start_point: tuple, end_point: tuple):
        self.start_point = start_point
        self.end_point = end_point
        self.mid_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
        self.id = id
        self.seen = False

class ShipObstacle:
    """
    Classe che rappresenta un ostacolo a forma di nave.
    Attributes:
        default_points (list): Punti del poligono della nave generati con i valori di default.
        points (list): Punti attuali del poligono della nave.
        x_max (float): Coordinata x massima del poligono.
        x_min (float): Coordinata x minima del poligono.
        y_max (float): Coordinata y massima del poligono.
        y_min (float): Coordinata y minima del poligono.
        center (tuple): Centro del poligono.
        radius (float): Raggio del poligono.
        segments_dict (dict): Dizionario dei segmenti del poligono.
    Methods:
        generate_segments() -> Dict:
            Genera i segmenti del poligono della nave.
        reset_ship() -> None:
            Resetta lo stato della nave ai valori di default
        rotate_ship(angle: float) -> None:
            Ruota il poligono della nave di un angolo (in radianti) specificato.
        translate_ship(vector: tuple) -> None:
            Trasla il poligono della nave di un vettore specificato.
        rototranslate_ship(angle: float, vector: tuple) -> None:
            Ruota e trasla il poligono della nave di un angolo e di un vettore specificati.
        random_placement(workspace: tuple, safe_distance: float) -> None:
            Genera la nave in un punto casuale del workspace e con una rotazione casuale.
        get_inflated_points(radius: float) -> list:
            Calcola i punti inflated del poligono della nave.
    """
    def __init__(self, ship_center: tuple, Options: Dict = {}, inflation_radius : float = 0.5, use_default_values: bool = True) -> None:
        """
        Inizializza un'istanza della classe.
        Args:
            ship_center (tuple): Coordinate del centro della nave.
            Options (Dict, optional): Dizionario contenente le opzioni per la generazione del poligono della nave. 
                                      Se non specificato, vengono utilizzati i valori di default. Default è {}.
            use_default_values (bool, optional): Se True, utilizza i valori di default per la generazione del poligono della nave. 
                                                 Se False, utilizza i valori presenti in Options. Default è True.
        Attributi:
            points (list): Lista dei punti che definiscono il poligono della nave.
            default_points (list): Copia dei punti di default del poligono della nave.
            x_max (float): Valore massimo delle coordinate x dei punti del poligono.
            x_min (float): Valore minimo delle coordinate x dei punti del poligono.
            y_max (float): Valore massimo delle coordinate y dei punti del poligono.
            y_min (float): Valore minimo delle coordinate y dei punti del poligono.
            default_limits (tuple): Limiti di default del poligono (x_max, x_min, y_max, y_min).
            center (tuple): Coordinate del centro del poligono.
            radius (float): Raggio massimo del poligono.
            segments_dict (dict): Dizionario contenente i segmenti del poligono.
            default_segments__dict (dict): Copia di default dei segmenti del poligono.
        """

        if use_default_values:
            L =                1
            W =                1.5
            a_left =           0.2
            a_right =          2.8
            points_distance =  0.2
            scale =            0.9
        else:
            L =                 Options['L']
            W =                 Options['W']
            a_left =            Options['a_left']
            a_right =           Options['a_right']
            points_distance =   Options['points_distance']
            scale =             Options['scale']
        
        self.inflation_radius = inflation_radius

        # Genera i punti del poligono della nave
        self.points = generate_ship_polygon(L, W, a_left, a_right, points_distance, ship_center, scale)
        # Salva i punti di default
        self.default_points = copy.deepcopy(self.points)
        
        # Puti inflated della ship
        self.inflated_points = self.get_inflated_points(self.inflation_radius)
        self.default_inflated_points = copy.deepcopy(self.inflated_points)

        # Estremi del poligono
        self.x_max = max([point[0] for point in self.points])
        self.x_min = min([point[0] for point in self.points])
        self.y_max = max([point[1] for point in self.points])
        self.y_min = min([point[1] for point in self.points])
        # Salva i limiti di default
        self.default_limits = (self.x_max, self.x_min, self.y_max, self.y_min)

        # Centro e raggio del poligono
        Cx = (self.x_max + self.x_min) / 2
        Cy = (self.y_max + self.y_min) / 2
        self.center = (Cx, Cy)
        max_radius = 0
        for point in self.points:
            dist = np.sqrt((point[0] - Cx)**2 + (point[1] - Cy)**2)
            if dist > max_radius:
                max_radius = dist
        self.radius = max_radius

        # Genera i segmenti
        self.segments_dict = self.generate_segments()
        # Salva i segmenti di default
        self.default_segments__dict = copy.deepcopy(self.segments_dict)

    def generate_segments(self)-> Dict:
        """
        Genera un dizionario di segmenti della nave collegando i punti definiti in self.points.
        Restituisce:
            Dict: Un dizionario dove le chiavi sono gli indici dei punti e i valori sono le istanze di ShipSegment che collegano i punti.
        """

        segments_dict = {}
        for i in range(len(self.points)):
            curr_point = self.points[i]

            # Se l'indice è l'ultimo, collega l'ultimo punto al primo
            if i == len(self.points) - 1:
                next_point = self.points[0]

            # Altrimenti, collega il punto corrente al punto successivo
            else:
                next_point = self.points[i + 1]
            
            # Crea un istanza di ShipSegment e lo memorizza nel dizionario
            segment = ShipSegment(i, curr_point, next_point)
            segments_dict[i] = segment

        return segments_dict
        
    def reset_ship(self) -> None:
        
        # Reset di tutti i punti della nave ai valori di default
        self.points = copy.deepcopy(self.default_points)

        # Reset dei punti inflated della nave
        self.inflated_points = copy.deepcopy(self.default_inflated_points)
    
        # Reset dei valori massimi e minimi delle coordinate x e y
        self.x_max, self.x_min, self.y_max, self.y_min = self.default_limits
        
        # Reset del centro della nave
        self.center = ((self.x_max + self.x_min) / 2, (self.y_max + self.y_min) / 2)

        # Reset dei segmenti della nave
        self.segments_dict = copy.deepcopy(self.default_segments__dict)
    
    def rotate_ship(self, angle: float) -> None:
        """
        Ruota la nave di un angolo specificato attorno al suo centro.
        Args:
            angle (float): L'angolo di rotazione in radianti.
        Effettua i seguenti passaggi:
        1. Calcola i nuovi punti ruotati della nave.
        2. Aggiorna i valori massimi e minimi delle coordinate x e y.
        3. Calcola i nuovi segmenti ruotati della nave.
        Returns:
            None
        """
        # Calcolo nuovi punti ruotati
        rot_points = []
        for point in self.points:
            point = rotate_point(point, self.center, angle)
            rot_points.append(point)
        self.points = rot_points
        
        # Calcolo nuovi punti inflated ruotati
        rot_inflated_points = []
        for point in self.inflated_points:
            point = rotate_point(point, self.center, angle)
            rot_inflated_points.append(point)
        self.inflated_points = rot_inflated_points

        # Aggiorna i valori massimi e minimi delle coordinate x e y
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
        """
        Trasla la nave di un vettore specificato.
        Questo metodo sposta tutti i punti della nave, il centro e i segmenti
        di un vettore dato. Aggiorna anche i limiti massimi e minimi delle
        coordinate x e y della nave.
        Args:
            vector (tuple): Un vettore (dx, dy) che specifica la traslazione
                    lungo gli assi x e y.
        Returns:
            None
        """

        dx, dy = vector

        # Traslo tutti i punti della nave
        translated_points = []
        for point in self.points:
            point = translate_point(point, vector)
            translated_points.append(point)
        self.points = translated_points
    	
        # Traslo tutti i punti inflated della nave
        inflated_translated_points = []
        for point in self.inflated_points:
            point = translate_point(point, vector)
            inflated_translated_points.append(point)
        self.inflated_points = inflated_translated_points

        # Aggiorno il centro e i valori massimi e minimi delle coordinate x e y
        self.center = (self.center[0] + dx, self.center[1] + dy)
        self.x_max += dx
        self.x_min += dx
        self.y_max += dy
        self.y_min += dy

        # Traslo tutti i segmenti della nave
        for segment in self.segments_dict.values():
            segment.start_point = translate_point(segment.start_point, vector)
            segment.end_point = translate_point(segment.end_point, vector)
            segment.mid_point = translate_point(segment.mid_point, vector)

    def rototranslate_ship(self, angle: float, vector: tuple) -> None:
        """
        Ruota e trasla la nave.
        Questa funzione ruota la nave di un angolo specificato e poi la trasla di una certa distanza.
        Args:
            angle (float): L'angolo in gradi di cui ruotare la nave.
            distance (tuple): Una tupla che rappresenta la distanza di traslazione (x, y).
        Returns:
            None
        """
        self.rotate_ship(angle)
        self.translate_ship(vector)

    def random_placement(self, workspace: tuple, safe_distance: float) -> None:
        """ 
        Funzione che genera la nave in un punto casuale del workspace e con una rotazione casuale.
        La nave viene generata in modo che si trovi ad una distanza di sicurezza dai bordi del workspace.
        """
        # Rotazione casuale
        self.rotate_ship(np.random.uniform(0, 2*np.pi))

        # Limiti del workspace
        x_min_ws, y_min_ws, x_max_ws, y_max_ws = workspace  
        # Lunghezza e larghezza del workspace
        Lx = x_max_ws - x_min_ws - 2*safe_distance
        Ly = y_max_ws - y_min_ws - 2*safe_distance
        
        # Limiti della nave
        x_max_ship = self.x_max
        x_min_ship = self.x_min
        y_max_ship = self.y_max
        y_min_ship = self.y_min
        # Lunghezza e larghezza della nave
        Lx_ship = x_max_ship - x_min_ship
        Ly_ship = y_max_ship - y_min_ship

        # Limiti del vector di traslazione
        x_max_vec = Lx/2 - Lx_ship/2
        y_max_vec = Ly/2 - Ly_ship/2
        
        # Se la nave è abbstanza piccola per essere traslata all'interno del workspace
        if x_max_vec > 0 and y_max_vec > 0:
            vec = (np.random.uniform(-x_max_vec, x_max_vec), np.random.uniform(-y_max_vec, y_max_vec))
            self.translate_ship(vec)

    def get_bounding_box(self) -> tuple:
        """	
        Restituisce i punti del bounding box della nave.
        Returns:
            tuple: Una tupla contenente i punti del bounding box (x_min, y_min, x_max, y_max).
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def get_edges(self) -> list:
        """
        Restituisce i segmenti della nave.
        Returns:
            list: Una lista di segmenti della nave.
        """
        edges = []
        for segment in self.segments_dict.values():
            edges.append((segment.start_point, segment.end_point))
        return edges

    def get_inflated_points(self, radius: float) -> list:
        """
        Calcola i punti inflated di un poligono dato un raggio specificato.
        Questo metodo prende i punti di un poligono e li trasla lungo la normale media
        dei lati adiacenti di una distanza specificata dal raggio. Il risultato è un
        nuovo set di punti che rappresentano il poligono inflated.
        Args:
            radius (float): La distanza di traslazione lungo la normale media per ogni punto del poligono.
        Returns:
            List[Tuple[float, float]]: Una lista di tuple che rappresentano i punti inflated del poligono.
        """

        inflated_points = []
        num_points = len(self.points)
        
        for i in range(num_points):
            # Ottieni i punti del vertice corrente, precedente e successivo
            prev_point = self.points[i - 1]
            curr_point = self.points[i]
            next_point = self.points[(i + 1) % num_points]
            
            # Calcola i vettori dei lati adiacenti
            edge1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
            edge2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
            
            # Calcola le normali esterne ai lati
            normal1 = (-edge1[1], edge1[0])
            normal2 = (-edge2[1], edge2[0])
            
            # Normalizza le normali
            length1 = (normal1[0]**2 + normal1[1]**2) ** 0.5
            length2 = (normal2[0]**2 + normal2[1]**2) ** 0.5
            normal1 = (normal1[0]/length1, normal1[1]/length1)
            normal2 = (normal2[0]/length2, normal2[1]/length2)
            
            # Calcola la normale media
            mean_normal = (normal1[0] + normal2[0], normal1[1] + normal2[1])
            length_mean = (mean_normal[0]**2 + mean_normal[1]**2) ** 0.5
            mean_normal = (mean_normal[0]/length_mean, mean_normal[1]/length_mean)
            
            # Trasla il vertice lungo la normale media
            inflated_point = (
                curr_point[0] + mean_normal[0] * radius,
                curr_point[1] + mean_normal[1] * radius
            )
            inflated_points.append(inflated_point)
        
        return inflated_points
    
    def point_in_ship(self, point: np.ndarray) -> bool:
        """
        Verifica se un punto si trova all'interno della nave.
        Questo metodo controlla se un punto specificato si trova all'interno del poligono
        rappresentato dai punti gonfiati della nave. Utilizza il prodotto vettoriale per
        determinare la posizione relativa del punto rispetto ai bordi del poligono.
        Args:
            point (array-like): Le coordinate del punto da verificare, sotto forma di array o lista.
        Returns:
            bool: True se il punto si trova all'interno del poligono, False altrimenti.
        """

        num_points = len(self.inflated_points)
        sign = None
        
        for i in range(num_points):
            p1 = np.array(self.inflated_points[i])
            p2 = np.array(self.inflated_points[(i + 1) % num_points])
            edge = p2 - p1
            to_point = point - p1
            cross = edge[0] * to_point[1] - edge[1] * to_point[0]
            
            current_sign = np.sign(cross)
            if current_sign == 0:
                continue  # Il punto è esattamente su un bordo
            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                return False  # Il punto è fuori
        return True  # Il punto è dentro
    
    def copy_segments(self) -> dict:
        """
        Restituisce una copia dei segmenti della nave.
        Returns:
            dict: Un dizionario contenente i segmenti della nave.
        """
        return copy.deepcopy(self.segments_dict)
    
def translate_point(point: tuple, vector: tuple) -> tuple:
    """
    Trasla un punto di una certa distanza.
    Args:
        point (tuple): Una tupla contenente le coordinate (x, y) del punto da traslare.
        r (tuple): Una tupla contenente le distanze (dx, dy) di traslazione lungo gli assi x e y.
    Returns:
        tuple: Una tupla contenente le nuove coordinate (x, y) del punto traslato.
    """

    x, y = point
    dx, dy = vector
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

def arc_points(a: float, b: float, center: tuple, theta_start: float, theta_end: float, points_distance: float, clockwise: bool = True, N_points: int = 50)-> tuple:
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
    num_points = int(np.ceil(total_arc_length / points_distance))

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

def generate_ship_polygon(L: float, W: float, a_left: float, a_right: float, points_distance: float, ship_center: tuple, scale: float = 1.0) -> list:
    """
    Genera un poligono che rappresenta una nave basato sui parametri forniti.
    Il poligono è composto da un rettangolo con due semicerchi ai lati e segmenti equidistanti sulla parte superiore e inferiore.
    Args:
        L (float): Lunghezza della parte rettilinea della nave.
        W (float): Larghezza della nave.
        a_left (float): Semiasse x dell'arco sinistro.
        a_right (float): Semiasse x dell'arco destro.
        points_distance (float): Distanza desiderata tra i punti del poligono.
        scale (float, opzionale): Fattore di scala per i parametri della nave. Default è 1.0.
    Returns:
        list: Una lista di tuple contenenti le coordinate (x, y) dei punti del poligono.
    Raises:
        ValueError: Se il numero di punti x e y non è lo stesso.
    """
    L = L * scale
    W = W * scale
    a_left = a_left * scale
    a_right = a_right * scale
    # points_distance = points_distance * scale

    b_left = W / 2
    b_right = W / 2

    center_left = (-L/2, 0)
    center_right = (L/2, 0)

    x_left, y_left = arc_points(a_left, b_left, center_left, -np.pi/2, np.pi/2, points_distance=points_distance, clockwise=True)
    x_right, y_right = arc_points(a_right, b_right, center_right, -np.pi/2, np.pi/2, points_distance=points_distance, clockwise=False)

    num_points_center = int(np.ceil(L / points_distance))

    x_center_top = np.linspace(-L/2, L/2, num_points_center)
    x_center_bottom = np.linspace(L/2, -L/2, num_points_center)
    y_center_top = np.full_like(x_center_top, W/2)
    y_center_bottom = np.full_like(x_center_bottom, -W/2)

    x = np.concatenate([x_left, x_center_top[1:-1], x_right, x_center_bottom[1:-1]])
    y = np.concatenate([y_left, y_center_top[1:-1], y_right, y_center_bottom[1:-1]])

    des_Cx, des_Cy = ship_center
    x_max = max(x)
    x_min = min(x)
    y_max = max(y)
    y_min = min(y)
    Cx = (x_max + x_min) / 2
    Cy = (y_max + y_min) / 2
    vector = (des_Cx - Cx, des_Cy - Cy)
    x = x + vector[0]
    y = y + vector[1]

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
    # points_distance = 0.2
    # points = generate_ship_polygon(L, W, a_left, a_right, points_distance)
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
    # Worksapce
    workspace = (0,0,8,8)
    workspace_center = ((workspace[0] + workspace[2]) / 2, (workspace[1] + workspace[3]) / 2)
    workspace_points = [(workspace[0], workspace[1]), (workspace[2], workspace[1]), (workspace[2], workspace[3]), (workspace[0], workspace[3]), (workspace[0], workspace[1])]
    x_ws, y_ws = zip(*workspace_points)
    
    # Genera la nave
    Ship = ShipObstacle(ship_center=workspace_center, use_default_values=True)
    
    Ship.random_placement(workspace, safe_distance=1.5)
    Ship.reset_ship()
    point = np.array([5.03,5.0])
    print(Ship.point_in_ship(point))

    edges = Ship.get_edges()
    print(len(edges))
    inflated_points = Ship.inflated_points
    x_inflate, y_inflate = zip(*inflated_points)
    # print(len(edges))
    # Ship.rototranslate_ship(0, (0,0))
    # Ship.reset_ship()
    Cx, Cy = Ship.center
    radius = Ship.radius
    points = Ship.points
    x, y = zip(*points)

    x_min = Ship.x_min
    x_max = Ship.x_max
    y_min = Ship.y_min
    y_max = Ship.y_max
    square_points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]
    x_square, y_square = zip(*square_points)

    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = Cx + radius * np.cos(theta)
    y_circle = Cy + radius * np.sin(theta)

    # Ship.rototranslate_ship(np.pi/4, (0,0))
    # rot_points = Ship.points
    # x_rot, y_rot = zip(*rot_points)
    
    # Ship.reset_ship()
    # points = Ship.points
    # x, y = zip(*points)

    dict_points = []
    for segment in Ship.segments_dict.values():
        x_c, y_c = segment.mid_point
        dict_points.append((x_c, y_c))
    x_c, y_c = zip(*dict_points)



    plt.figure(figsize=(8, 4))
    plt.plot(x, y, '-o', label='Rettangolo con Semicerchi e Segmenti Equidistanti')
    # plt.plot(x_rot, y_rot, '-o', label='Rettangolo con Semicerchi e Segmenti Equidistanti Ruotato di 45°')
    plt.plot(x_inflate, y_inflate, '-o', label='Rettangolo Inflato')
    plt.scatter(Cx, Cy, color='r', label='Centro della Nave', s=100)
    # plt.plot(x_square, y_square, '-o', label='Quadrato di Bounding Box')
    plt.scatter(point[0], point[1], color='g', label='Punto da Testare', s=100)

    plt.plot(x_c, y_c, '+r', label='Punti Medi dei Segmenti')
    # plt.plot(x_circle, y_circle, '--', label='Circonferenza di Bounding Box')
    plt.axis('equal')
    plt.title('Approssimazione della Chiglia con Segmenti Equidistanti sui Lati Orizzontali')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()
    # for i in range(5):
    #     # vec = (np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5))
    #     # theta = np.random.uniform(0, 2*np.pi)
    #     # Ship.rototranslate_ship(theta, vec)
    #     # Ship.translate_ship(vec)
    #     Ship.random_placement(workspace, safe_distance=1.5)
    #     # Ship.translate_ship((-1.5, 1.5))
    #     points = Ship.points
    #     x, y = zip(*points)

    #     dict_points = []
    #     for segment in Ship.segments_dict.values():
    #         x_c, y_c = segment.mid_point
    #         dict_points.append((x_c, y_c))
    #     x_c, y_c = zip(*dict_points)

    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     ax.set_xlim(0,8)
    #     ax.set_ylim(0,8)
    #     ax.plot(x_ws, y_ws, '-k', label='Workspace')
    #     ax.plot(x, y, '-o', label='Ship')
    #     ax.plot(x_c, y_c, '+r', label='Mid Points')
    #     plt.grid(True)
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.show()

    #     Ship.reset_ship()