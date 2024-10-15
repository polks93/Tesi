import numpy as np
import warnings
from my_package.core import lidar, discretize_lidar

from typing import Tuple, Union, Sequence

class Unicycle():
    def __init__(self, init_pose=[0,0,0], footprint={}, lidar_parms={'FoV': np.deg2rad(60), 'max_range': 3.0, 'n_beams': 10}):
        """ Costruttore per la classe Unicycle """

        # Inizializza lo stato dell'uniciclo
        self.init_pose = init_pose
        self.reset()

        # Definisce le velocità massime lineari e angolari
        self.max_v      = 0.75
        self.max_omega  = 2.5

        # Definisce la footprint dell'uniciclo
        if not isinstance(footprint, dict):
            warnings.warn("footprint non definita. Imposto la footprint puntiforme.")
            footprint = {}
        elif 'radius' not in footprint and 'square' not in footprint:
            warnings.warn("footprint non definita. Imposto la footprint puntiforme.")
            footprint = {}
        self.footprint = footprint

        # Definisce i parametri del sensore LIDAR
        self.lidar_params = lidar_parms

    def reset(self, init_pose=None):
        """
        Resetta lo stato dell'uniciclo.
        Questo metodo imposta la posizione (x, y), l'orientamento (theta), la velocità lineare (v) e la velocità angolare (omega) dell'uniciclo
        ai loro valori iniziali.
        """
        if init_pose is None:
            init_pose = self.init_pose
        else:
            self.init_pose = init_pose
            
        self.x      = self.init_pose[0]
        self.y      = self.init_pose[1]
        self.theta  = self.init_pose[2]
        self.v      = 0
        self.omega  = 0

    def incremental_kinematics(self, dv=0.0, domega=0.0, dt=1.0):
        """
        Aggiorna lo stato del modello di uniciclo in base ai cambiamenti forniti di velocità e velocità angolare.
        Parametri:
            dv (float): Cambiamento nella velocità lineare.
            domega (float): Cambiamento nella velocità angolare.
        Questa funzione modifica la velocità attuale e la velocità angolare dell'uniciclo, assicurandosi che rimangano entro i limiti specificati. 
        Successivamente aggiorna la posizione (x, y) e l'orientamento (theta) dell'uniciclo in base alla nuova velocità e velocità angolare.
        """
        # Aggiorna la velocità e la velocità angolare
        self.v      += dv
        self.omega  += domega
        # Limita la velocità e la velocità angolare per rimanere entro i limiti specificati
        self.v      = np.clip(self.v, -self.max_v, self.max_v)
        self.omega  = np.clip(self.omega, -self.max_omega, self.max_omega)
        # Aggiorna la posizione e l'orientamento dell'uniciclo
        self.x      += self.v * np.cos(self.theta) * dt
        self.y      += self.v * np.sin(self.theta) * dt
        self.theta  = self.wrapToPi(self.theta + self.omega * dt)
        
    def simple_kinematics(self, omega, dt=1.0):
        """
        Aggiorna la posizione del robot uniciclo utilizzando la cinematica semplice.
        Parametri:
        - omega (float): Velocità angolare del robot.
        - dt (float): Intervallo di tempo per l'aggiornamento.
        Ritorna:
        Nessuno
        """
        self.omega = omega
        self.v = self.max_v
        self.theta = self.wrapToPi(self.theta + self.omega * dt)
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt


    def wrapToPi(self, angle):
        """
        Converte l'angolo di input nell'intervallo [-pi, pi].
        Parametri:
            angle (float): L'angolo di input da convertire.
        Ritorna:
            float: L'angolo convertito nell'intervallo [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def get_state(self) -> np.ndarray:
        """
        Restituisce lo stato attuale dell'uniciclo.
        Ritorna:
            np.array: Un array contenente la posizione (x, y), l'orientamento (theta), la velocità lineare (v) e la velocità angolare (omega).
        """
        return np.array([self.x, self.y, self.theta, self.v, self.omega])
    
    def set_state(self, state: np.ndarray) -> None:
        """
        Imposta lo stato dell'uniciclo.
        Parametri:
            state (np.array): Un array contenente la posizione (x, y), l'orientamento (theta), la velocità lineare (v) e la velocità angolare (omega).
        """
        self.x, self.y, self.theta, self.v, self.omega = state


    def collision_check(self, obstacle: Sequence) -> bool:
        """
        Controlla le collisioni tra la footprint dell'oggetto e una lista di ostacoli.
        Questa funzione itera attraverso una lista di ostacoli e verifica se qualche punto 
        della footprint dell'oggetto interseca con il rettangolo di delimitazione degli ostacoli. 
        La posizione e l'orientamento dell'oggetto vengono presi in considerazione per calcolare le 
        coordinate effettive dei punti della footprint.
        Parametri:
            obstacles (list): Una lista di ostacoli, dove ogni ostacolo è definito 
                              da una tupla di quattro valori (xmin, ymin, xmax, ymax) 
                              che rappresentano il rettangolo di delimitazione.
        Ritorna:
            bool: True se viene rilevata una collisione, False altrimenti.
        """
        # footprint circolare
        if 'radius' in self.footprint:  
            radius = self.footprint['radius']
            xmin, ymin, xmax, ymax = obstacle
            xmin = xmin - radius
            ymin = ymin - radius
            xmax = xmax + radius
            ymax = ymax + radius

            if xmin < self.x < xmax and ymin < self.y < ymax:
                return True


        # footprint quadrata       
        elif 'square' in self.footprint:  
            for point in self.footprint['square']:
                x = self.x + point[0] * np.cos(self.theta) - point[1] * np.sin(self.theta)
                y = self.y + point[0] * np.sin(self.theta) + point[1] * np.cos(self.theta)
                if obstacle[0] < x < obstacle[2] and obstacle[1] < y < obstacle[3]:
                    return True
        # Nessuna footprint definita 
        else:
            if obstacle[0] < self.x < obstacle[2] and obstacle[1] < self.y < obstacle[3]:
                return True
        return False
    
    def boundary_check(self, workspace: Sequence) -> bool:
        """
        Controlla se tutti i punti della footprint dell'oggetto sono all'interno del workspace specificato.
        Questa funzione itera attraverso i punti della footprint e verifica se ciascun punto 
        è all'interno del rettangolo di delimitazione definito dal workspace.
        Parametri:
            workspace (tuple): Una tupla di quattro valori (xmin, ymin, xmax, ymax) 
                               che rappresentano il rettangolo di delimitazione del workspace.
        Ritorna:
            bool: True se tutti i punti sono all'interno del workspace, False altrimenti.
        """
        # footprint circolare
        if 'radius' in self.footprint:
            if (workspace[0] + self.footprint['radius'] <= self.x <= workspace[2] - self.footprint['radius'] and
                workspace[1] + self.footprint['radius'] <= self.y <= workspace[3] - self.footprint['radius']):
                return True
            return False
            
        # footprint quadrata
        elif 'square' in self.footprint: 
            for point in self.footprint['square']:
                x = self.x + point[0] * np.cos(self.theta) - point[1] * np.sin(self.theta)
                y = self.y + point[0] * np.sin(self.theta) + point[1] * np.cos(self.theta)
                if not (workspace[0] <= x <= workspace[2] and workspace[1] <= y <= workspace[3]):
                    return False
            return True
        
        # Nessuna footprint definita
        else:
            if workspace[0] <= self.x <= workspace[2] and workspace[1] <= self.y <= workspace[3]:
                return True
            return False


    def lidar(self, obstacle: Sequence) -> np.ndarray:
        """
        Calcola e restituisce le distanze misurate dal sensore LIDAR rispetto agli ostacoli presenti nell'ambiente.
        Args:
            obstacle (Sequence): Una sequenza di quattro valori (xmin, ymin, xmax, ymax) che rappresentano il rettangolo di delimitazione dell'ostacolo.
        Returns:
            np.array: Lista delle distanze misurate dal sensore LIDAR rispetto agli ostacoli.
        """
        
        pose = self.get_state()[:3]
        ranges = lidar(pose, obstacle, self.lidar_params)
        return ranges
    
    def discrete_lidar(self, obstacle: Sequence, safe_distance: float, sector_indices: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Esegue la discretizzazione dei dati del lidar in settori e determina se c'è un allarme di sicurezza.
        Args:
            obstacle (Sequence): Una sequenza di quattro valori (xmin, ymin, xmax, ymax) che rappresentano il rettangolo di delimitazione dell'ostacolo.
            safe_distance (float): Distanza di sicurezza entro la quale viene generato un allarme.
            sector_indices (np.ndarray): Indici dei settori in cui dividere i dati del lidar.
        Returns:
            Tuple[np.ndarray, bool]: Una tupla contenente un array con i settori discretizzati e un booleano che indica se c'è un allarme di sicurezza.
        """

        pose = self.get_state()[:3]
        sectors, alert = discretize_lidar(pose, self.lidar_params, obstacle, safe_distance, sector_indices)

        return sectors, alert   

if __name__ == "__main__":

    footprint = {'radius': 0.15}

    lidar_params = {
        'max_range': 1.6,
        'n_beams': 10,
        'FoV': np.deg2rad(60)
    }
    # Crea un'istanza della classe Unicycle
    unicycle = Unicycle(init_pose=[0, 0, np.pi/4], footprint=footprint, lidar_parms=lidar_params)
    
    # Esempio di aggiornamento dello stato
    
    # Ottieni lo stato attuale
    state = unicycle.get_state()
    print("Stato attuale:", state)
    
    # Definisco un ostacolo
    obstacle = (2, 2, 3, 3)
    
    # Controlla le collisioni
    collision = unicycle.collision_check(obstacle)
    print("Collisione rilevata:", collision)
    
    # Definisci i limiti del workspace
    workspace = [0, 0, 5, 5]
    
    # Controlla se è all'interno dei limiti
    within_bounds = unicycle.boundary_check(workspace)
    print("All'interno dei limiti del workspace:", within_bounds)

    ranges = unicycle.lidar(obstacle)
    sectors, alert = unicycle.discrete_lidar(obstacle, safe_distance=0.5, sector_indices=np.array([[0, 3], [4, 7], [8, 9]]))
    print(ranges)