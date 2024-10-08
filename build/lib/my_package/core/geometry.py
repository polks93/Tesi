import math
import numpy as np
from typing import Sequence, Union, Tuple, Dict, Optional
import matplotlib.pyplot as plt

class Segment:
    """
    Classe che rappresenta un segmento con un punto iniziale e un punto finale, 
    un identificatore e un lato valido.
    Attributi:
    ----------
    id : int
        Identificatore univoco del segmento.
    start_point : Union[Sequence, np.ndarray]
        Punto iniziale del segmento.
    end_point : Union[Sequence, np.ndarray]
        Punto finale del segmento.
    mid_point : tuple
        Punto medio del segmento, calcolato come la media dei punti iniziale e finale.
    side : str
        Lato del segmento, deve essere uno dei lati validi definiti in `metadata`.
    seen : bool
        Flag per indicare se il segmento è stato visto o meno.
    """

    metadata = {'valid_sides': ['top', 'bottom', 'left', 'right']}

    def __init__(self, id: int, start_point: Union[Sequence, np.ndarray], end_point: Union[Sequence, np.ndarray], side: str, neighbor: Optional[str] = None):
        if side not in self.metadata['valid_sides']:
            raise ValueError(f"Invalid side: {side}. Valid sides are: {self.metadata['valid_sides']}")
        
        self.id = id
        self.start_point = start_point
        self.end_point = end_point
        self.mid_point = ((start_point[0] + end_point[0]) / 2,
                            (start_point[1] + end_point[1]) / 2)
        self.side = side
        self.neighbor = neighbor
        self.seen = False

def generate_segments_for_side(
        start_point: Union[Sequence, np.ndarray], 
        end_point: Union[Sequence, np.ndarray],
        side: str,
        segment_id: int,
        segment_length: float
    ) -> Tuple[dict, int]:

    """
    Genera segmenti per un lato specificato tra due punti.
    Args:
        start_point (Union[Sequence, np.ndarray]): Il punto di inizio del lato.
        end_point (Union[Sequence, np.ndarray]): Il punto di fine del lato.
        side (str): Il lato per il quale generare i segmenti.
        segment_id (int): L'ID iniziale per i segmenti generati.
        segment_length (float): La lunghezza desiderata per ciascun segmento.
    Returns:
        Tuple[dict, int]: Un dizionario contenente i segmenti generati e l'ID successivo disponibile.
    """

    start_point = np.array(start_point)
    end_point = np.array(end_point)
    side_length = np.linalg.norm(end_point - start_point)
    num_segments = int(np.ceil(side_length / segment_length))

    match(side):
        case 'bottom':
            start_neighbor = 'left'
            end_neighbor = 'right'
        case 'top':
            start_neighbor = 'left'
            end_neighbor = 'right'
        case 'left':
            start_neighbor = 'bottom'
            end_neighbor = 'top'
        case 'right':
            start_neighbor = 'bottom'
            end_neighbor = 'top'
        case _:
            raise ValueError(f"Invalid side: {side}. Valid sides are: {Segment.metadata['valid_sides']}")

    # Aggiusto la lunghezze dei segmenti in caso length non sia un multiplo di segment_length
    if num_segments > 0:
        segment_length = float(side_length) / num_segments
    else:
        segment_length = float(side_length)
        num_segments = 1
    
    # Genero i punti su tutta la lunghezza del lato
    points = np.linspace(start_point, end_point, num=num_segments + 1)
    segments = {}
    id = segment_id

    for i in range(num_segments):
        sp = points[i]  
        ep = points[i + 1]

        if i == 0:
            neighbor = start_neighbor
        elif i == num_segments - 1:
            neighbor = end_neighbor
        else:
            neighbor = None

        segment = Segment(id=id, start_point=sp, end_point=ep, side=side, neighbor=neighbor)
        segments[id] = segment
        id += 1
    return segments, id

def generate_segments(obstacle: Sequence, segment_length: float) -> Dict[int, Segment]:
    """
    Genera segmenti che dividono ogni lato di un ostacolo rettangolare.
    Args:
        obstacle (Sequence): Una sequenza contenente le coordinate del rettangolo (xmin, ymin, xmax, ymax).
        segment_length (float): La lunghezza di ciascun segmento da generare.
    Returns:
        Dict[int, Segment]: Un dizionario dove le chiavi sono gli ID dei segmenti e i valori sono i segmenti stessi.
    """

    xmin, ymin, xmax, ymax = obstacle
    segments_dict = {}
    id = 0

    # Bottom side
    start_point = (xmin, ymin)
    end_point = (xmax, ymin)
    segments, id = generate_segments_for_side(start_point, end_point, 'bottom', id, segment_length)
    segments_dict.update(segments)

    # Top side
    start_point = (xmin, ymax)
    end_point = (xmax, ymax)
    segments, id = generate_segments_for_side(start_point, end_point, 'top', id, segment_length)
    segments_dict.update(segments)

    # Left side
    start_point = (xmin, ymin)
    end_point = (xmin, ymax)
    segments, id = generate_segments_for_side(start_point, end_point, 'left', id, segment_length)
    segments_dict.update(segments)

    # Right side
    start_point = (xmax, ymin)
    end_point = (xmax, ymax)
    segments, id = generate_segments_for_side(start_point, end_point, 'right', id, segment_length)
    segments_dict.update(segments)

    return segments_dict

def distance(P1: Union[Sequence, np.ndarray], P2: Union[Sequence, np.ndarray]) -> np.float64:
    """ Distanza tra due punti """
    P1 = np.array(P1)
    P2 = np.array(P2)
    return np.linalg.norm(P1 - P2)

def heading(P1: Union[Sequence, np.ndarray], P2: Union[Sequence, np.ndarray]) -> float:
    return math.atan2(P2[1] - P1[1], P2[0] - P1[0])

def is_in_FoV(agent_pos: Union[Sequence, np.ndarray], FoV: float, range: float, P: Sequence) -> bool:
    """
    Verifica se un punto è all'interno del campo visivo di un agente.
    Args:
        agent_pos (Sequence): Le coordinate (x, y, theta) dell'agente.
        FoV (float): L'ampiezza del campo visivo dell'agente in radianti.
        range (float): La distanza massima a cui l'agente può rilevare il punto.
        P (Sequence): Le coordinate (x, y) del punto da verificare.
    Returns:
        bool: True se il punto è all'interno del campo visivo, False altrimenti.
    """
    
    x, y, theta = agent_pos

    # Verifica se il punto è all'interno del raggio
    if distance((x, y), P) > range:
        return False
    
    # Calcolo l'angolo tra l'agente e il punto
    angle_to_point = heading(agent_pos, P)
    # Calcola la differenza angolare
    angle_diff = angle_to_point - theta

    # Normalizza la differenza angolare nell'intervallo [-pi, pi]
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    # Verifica se il punto è all'interno del campo visivo
    if - FoV / 2 <= angle_diff <= FoV / 2:
        return True
    else:
        return False

def is_in_rect(P: Union[Sequence, np.ndarray], rect: Sequence) -> bool:
    """
    Verifica se il punto P è all'interno del rettangolo definito dai punti xmin, ymin, xmax e ymax.
    Args:
        P (Sequence): Le coordinate del punto da verificare.
        rect (Sequence): Le coordinate del rettangolo nel formato (xmin, ymin, xmax, ymax).
    Returns:
        bool: True se il punto è all'interno del rettangolo, False altrimenti.
    """
    if rect[0] <= P[0] <= rect[2] and rect[1] <= P[1] <= rect[3]:
        return True
    else:
        return False
    
def point_to_segment_distance(P: np.ndarray, s: Sequence[np.ndarray]) -> np.float64:
    """
    Calcola la distanza minima tra un punto e un segmento.
    Parametri:
    P (np.ndarray): Coordinata del punto.
    s (np.ndarray): Segmento definito da due punti.
    Ritorna:
    np.float64: La distanza minima tra il punto e il segmento.
    """
    # Estrai i punti del segmento
    v = s[0]
    w = s[1]

    # Calcola la proiezione del punto P sulla linea del segmento
    l2 = np.sum((w - v) ** 2)  # Lunghezza al quadrato del segmento
    if l2 == 0.0:
        return np.linalg.norm(P - v)  # s[0] e s[1] sono lo stesso punto
    t = max(0, min(1, np.dot(P - v, w - v) / l2))
    projection = v + t * (w - v)
    return np.linalg.norm(P - projection)

def is_rect_in_max_range(rectangle: Sequence, pose: np.ndarray, max_range: float) -> bool:
    """
    Verifica se un rettangolo è all'interno di un raggio massimo da una data posizione.
    Args:
        rectangle (Tuple): Coordinate del rettangolo nel formato (xmin, ymin, xmax, ymax).
        pose (Union[Sequence, np.ndarray]): Posizione di riferimento nel formato (cx, cy).
        max_range (float): Raggio massimo entro il quale verificare la presenza del rettangolo.
    Returns:
        bool: True se almeno un segmento del rettangolo è all'interno del raggio massimo, False altrimenti.
    """
    # Coordinate del centro del raggio
    C = pose[:2]
    # Coordinate del rettangolo
    xmin, ymin, xmax, ymax = rectangle
    # Coordinate di tutti i vertici
    vertices = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    # Segmenti del rettangolo
    segments = [(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    # Verifica se almeno un segmento è all'interno del raggio massimo
    for s in segments:
        if point_to_segment_distance(C, s) <= max_range:
            return True
    return False

def find_sectors_indices(n_beams: int, n_sectors: int) -> np.ndarray:
    """
    Trova gli indici dei settori dati il numero di fasci e il numero di settori.
    Args:
        n_beams (int): Il numero totale di fasci.
        n_sectors (int): Il numero totale di settori.
    Returns:
        np.ndarray: Un array di indici di forma (n_sectors, 2) dove ogni riga rappresenta
                    gli indici di inizio e fine di un settore.
    """
    indices = np.zeros((n_sectors,2), dtype=int)    
    for i in range(n_sectors):
        indices[i][0] = math.ceil(i * ((n_beams  - 1)/ 3))
        indices[i][1] = math.floor((i + 1) * (n_beams - 1)/3 + 1)
    return indices

def generate_random_obstacle(perimeter: int, workspace: Sequence, safe_distance: int = 2) -> Sequence:	
    """
    Genera un ostacolo rettangolare casuale all'interno di un'area di lavoro specificata, rispettando una distanza di sicurezza dai bordi.
    Args:
        perimeter (int): Il perimetro dell'ostacolo rettangolare da generare.
        workspace (Sequence): Una sequenza di quattro valori che definiscono i limiti dell'area di lavoro (xmin, ymin, xmax, ymax).
        safe_distance (int, opzionale): La distanza di sicurezza dai bordi dell'area di lavoro. Default è 2.
    Returns:
        Sequence: Una sequenza di quattro valori che definiscono i limiti dell'ostacolo rettangolare generato (rect_xmin, rect_ymin, rect_xmax, rect_ymax).
    Raises:
        ValueError: Se l'area di lavoro è troppo piccola per applicare la distanza di sicurezza.
    """

    xmin, ymin, xmax, ymax = workspace
    
    # Limiti dell'area di lavoro con la safe_distance
    safe_xmin = xmin + safe_distance
    safe_ymin = ymin + safe_distance
    safe_xmax = xmax - safe_distance
    safe_ymax = ymax - safe_distance

    if safe_xmin >= safe_xmax or safe_ymin >= safe_ymax:
        raise ValueError("L'area di lavoro è troppo piccola per applicare la distanza di sicurezza.")

    while True:
        height = np.random.randint(1, (perimeter // 2))
        width = (perimeter // 2) - height
        
        if width <= (safe_xmax - safe_xmin) and height <= (safe_ymax - safe_ymin):
            break

    rect_xmin = np.random.randint(safe_xmin, safe_xmax - width + 1)
    rect_ymin = np.random.randint(safe_ymin, safe_ymax - height + 1)

    rect_xmax = rect_xmin + width
    rect_ymax = rect_ymin + height

    return (rect_xmin, rect_ymin, rect_xmax, rect_ymax)

def segments_intersect(segment1: Union[Sequence, np.ndarray], segment2: Union[Sequence, np.ndarray]) -> bool:
    """
    Verifica se due segmenti si intersecano.
    Args:
        segment1 (Union[Sequence, np.ndarray]): Il primo segmento, definito da due punti (P1, P2).
        segment2 (Union[Sequence, np.ndarray]): Il secondo segmento, definito da due punti (P3, P4).
    Returns:
        bool: True se i segmenti si intersecano, False altrimenti.
    Raises:
        ValueError: Se uno dei segmenti non è definito da esattamente due punti.
    """
    if len(segment1) != 2 or len(segment2) != 2:
        if len(segment1[0]) != 2 or len(segment1[1]) != 2 or len(segment2[0]) != 2 or len(segment2[1]) != 2:
            raise ValueError("I segmenti devono essere definiti da due punti ciascuno.")
        
    x1, y1 = segment1[0]
    x2, y2 = segment1[1]
    x3, y3 = segment2[0]
    x4, y4 = segment2[1]

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-8:
        return False
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        return True
    else:
        return False

def is_segment_visible(pose: Union[Sequence, np.ndarray], segment: Segment, obstacle: Sequence, lidar_parameters: Dict) -> bool:

    # Verifica se il segmento è nel campo visivo
    if not is_in_FoV(pose, lidar_parameters['FoV'], lidar_parameters['max_range'], segment.mid_point):
        return False

    # Elimino dai lati da controllare quelli in cui si trova il segmento
    segment_side = segment.side
    obstacle_sides = ['bottom', 'top', 'left', 'right']
    obstacle_sides.remove(segment_side)

    # Linea tra il punto di osservazione e il punto medio del segmento
    line_of_sight = (pose[:2], segment.mid_point)

    # Cerco un intersezione tra la linea di vista e gli altri lati dell'ostacolo
    for side in obstacle_sides:
        match(side):
            case 'bottom':
                P1 = (obstacle[0], obstacle[1])
                P2 = (obstacle[2], obstacle[1])
            case 'top':
                P1 = (obstacle[0], obstacle[3])
                P2 = (obstacle[2], obstacle[3])
            case 'left':
                P1 = (obstacle[0], obstacle[1])
                P2 = (obstacle[0], obstacle[3])
            case 'right':
                P1 = (obstacle[2], obstacle[1])
                P2 = (obstacle[2], obstacle[3])
            case _:
                raise ValueError(f"Invalid side: {side}. Valid sides are: {obstacle_sides}")
        if segments_intersect(line_of_sight, (P1, P2)):
            return False

    return True                
            

if __name__ == "__main__":
    """ Test line of sigth """
    # pose = (0, 1.2, 0)
    # obstacle = (1, 1, 2, 2)
    # segment = Segment(0, (1, 1), (1.5, 1), 'bottom')
    # lidar_parameters = {'FoV': np.deg2rad(60), 'max_range': 1.5}
    # print(is_segment_visible(pose, segment, obstacle, lidar_parameters))  # True

    """Test generate_segments_for_side"""
    obstacle = [0, 0, 2, 2]
    segment_length = 0.5
    segments = generate_segments(obstacle, segment_length)
    # segments = generate_segments(obstacle, segment_length)

    for id, segment in segments.items():
        print(f"Segment {id}: {segment.start_point} -> {segment.end_point} ({segment.side}), neighbor: {segment.neighbor}")

    """Test intersezione segmenti """
    # segment1 = [(-1, -1), (2, 2)]
    # segment2 = [(0, 2), (2, 4)]

    # print(intersection_between_segments(segment1, segment2))  # True

    # Test the functions
    # P = np.array([0.0, 0.0, 0.0])
    # FoV = np.deg2rad(60)
    # rect = [1.5,-1,3,1]
    # print(is_rect_in_max_range(rect, P, 1.0))  # True
    # n_beams = 10
    # n_sectors = 3s
    # vec = find_sectors_indices(n_beams, n_sectors)
    # print(vec)
    """" Test ostacolo random """
    # workspace = (0, 0, 8, 8)
    # safe_distance = 2
    # obstacle_perimeter = 12
    # for i in range(10):
    #     rectangle = generate_random_obstacle(obstacle_perimeter, workspace, safe_distance)
    #     print(rectangle)
        
    #     # Estrai i limiti dall'area di lavoro e dal rettangolo
    #     wxmin, wymin, wxmax, wymax = workspace
    #     rxmin, rymin, rxmax, rymax = rectangle

    #     # Crea la figura e l'asse
    #     fig, ax = plt.subplots()

    #     # Imposta i limiti dell'area di lavoro
    #     ax.set_xlim(wxmin, wxmax)
    #     ax.set_ylim(wymin, wymax)

    #     # Disegna le 4 linee del rettangolo
    #     # Lato inferiore
    #     ax.plot([rxmin, rxmax], [rymin, rymin], color='r', linewidth=2)
    #     # Lato superiore
    #     ax.plot([rxmin, rxmax], [rymax, rymax], color='r', linewidth=2)
    #     # Lato sinistro
    #     ax.plot([rxmin, rxmin], [rymin, rymax], color='r', linewidth=2)
    #     # Lato destro
    #     ax.plot([rxmax, rxmax], [rymin, rymax], color='r', linewidth=2)

    #     # Etichetta degli assi e titolo
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_title('Perimetro del rettangolo all\'interno dell\'area di lavoro')

    #     # Mostra la griglia e il grafico
    #     ax.grid(True)
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.show()