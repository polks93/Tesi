import math
import numpy as np
from typing import Sequence, Union
import numpy.typing as npt

def distance(P1, P2) -> float:
    return math.sqrt((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2)

def heading(P1, P2) -> float:
    return math.atan2(P2[1] - P1[1], P2[0] - P1[0])

def is_in_FoV(agent_pos: Union[Sequence, npt.NDArray], FoV: float, range: float, P: Sequence) -> bool:
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

def is_in_rect(P: Sequence, rect: Sequence) -> bool:
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
    
if __name__ == "__main__":
    # Test the functions
    point = (10, 1)
    rect = (0,0,2,2)
    print(is_in_rect(point, rect))  
