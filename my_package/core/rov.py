import numpy as np
import warnings
from typing import Tuple, Sequence, Optional

from my_package.core import ShipObstacle
from my_package.core.ship_detection import lidar

class Rov:
    def __init__(
            self, 
            init_pose: np.ndarray = np.array([0,0,0]), 
            footprint: dict ={}, 
            lidar_parms: dict ={'FoV': np.deg2rad(60), 'max_range': 3.0, 'n_beams': 10}
            ) -> None:
        
        # Inizializzo lo stato del ROV
        self.init_pose = init_pose
        self.reset()

        # Definisce le velocità massime (modello uniciclo)
        self.max_v      = 0.75
        self.max_omega  = 2.5

        # Definise le velocità massime (modello di Zeno)
        self.max_v_zeno = 0.2
        self.max_omega_zeno = 0.15

        # Definisce la footprint dell'uniciclo
        if 'radius' not in footprint:
            warnings.warn("footprint non definita correttamente. Imposto la footprint circolare con raggio 0.5.")
            footprint = {'radius': 0.5}
        self.footprint = footprint

        # Definisce i parametri del sensore LIDAR
        self.lidar_params = lidar_parms


    def reset(self, init_pose: Optional[np.ndarray] = None) -> None:
        """
        Resetta lo stato dell'uniciclo.
        Questo metodo imposta la posizione (x, y), l'orientamento (theta), la velocità lineare (v) e la velocità angolare (omega) dell'uniciclo
        ai loro valori iniziali.
        """
        if init_pose is not None:
            self.init_pose = init_pose

        self.x = self.init_pose[0]
        self.y = self.init_pose[1]
        self.theta = self.init_pose[2]
        self.v = 0.0
        self.omega = 0.0
        self.vx = 0.0
        self.vy = 0.0

    def saturate(self, value: float, max_value: float) -> Tuple[float, bool]:
        """
        Limita il valore assoluto di un numero al massimo specificato.
        Args:
            value (float): Il valore da limitare.
            max_value (float): Il valore massimo assoluto consentito.
        Returns:
            float: Il valore limitato
        """

        if value > max_value:
            return max_value, True
        elif value < - max_value:
            return - max_value, True
        return value, False
    
    def zeno_kinematics(self, dvx: float, dvy: float, domega: float, dt: float=1.0) -> list[bool]:
        """
        Muove il ROV usando la cinematica di Zeno.
        Args:
            - dvx (float): Variazione della velocità lungo l'asse x.
            - dvy (float): Variazione della velocità lungo l'asse y.
            - domega (float): Variazione della velocità angolare.
            - dt (float): Intervallo di tempo per l'aggiornamento.
        Returns:    
            None
        """
        saturation = [False, False, False]
        self.vx, saturation[0] = self.saturate(self.vx + dvx, self.max_v_zeno)
        self.vy, saturation[1] = self.saturate(self.vy + dvy, self.max_v_zeno)
        self.omega, saturation[2] = self.saturate(self.omega + domega, self.max_omega_zeno)

        self.theta = self.wrapToPi(self.theta + self.omega * dt)
        self.x += (self.vx * np.cos(self.theta)  - self.vy * np.sin(self.theta)) * dt
        self.y += (self.vx * np.sin(self.theta)  + self.vy * np.cos(self.theta)) * dt

        return saturation

    def unicycle_kinematics(self, omega: float, dt: float=1.0):
        """
        Muove il ROV usando la cinematica dell'uniciclo.
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

    def wrapToPi(self, angle: float):
        """
        Converte l'angolo di input nell'intervallo [-pi, pi].
        Parametri:
            angle (float): L'angolo in radianti da convertire.
        Ritorna:
            float: L'angolo convertito nell'intervallo [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def get_state(self) -> np.ndarray:
        """
        Restituisce lo stato attuale del ROV.
        Returns:
            np.array: Un array contenente la posizione (x, y), l'orientamento (theta), la velocità lineare (v) e la velocità angolare (omega).
        """
        return np.array([self.x, self.y, self.theta, self.v, self.omega])
    
    def get_zeno_state(self) -> np.ndarray:
        """
        Restituisce lo stato attuale del ROV.
        Returns:
            np.array: Un array contenente la posizione (x, y), l'orientamento (theta), la velocità lineare (v) e la velocità angolare (omega).
        """
        return np.array([self.x, self.y, self.theta, self.vx, self.vy, self.omega])
    
    def collision_check(self, Ship: ShipObstacle) -> bool:
        """
        Controlla se il ROV è in collisione con un ostacolo definito da un oggetto ShipObstacle.
        Questo metodo verifica se il ROV si sovrappone all'ostacolo Ship.
        Parametri:
            Ship (ShipObstacle): L'ostacolo da controllare.
        Returns:
            bool: True se il ROV è in collisione con l'ostacolo, False altrimenti.
        """
        Cx_ship, Cy_ship = Ship.center
        position = self.get_state()[:2]
        x, y = position
        if np.linalg.norm([x - Cx_ship, y - Cy_ship]) > self.footprint['radius'] + Ship.radius:
            return False
        
        else:
            return Ship.point_in_ship(point=position)

    def boundary_check(self, workspace: Sequence) -> bool:
        """
        Controlla se tutti i punti della footprint dell'oggetto sono all'interno del workspace specificato.
        Questa funzione itera attraverso i punti della footprint e verifica se ciascun punto 
        è all'interno del rettangolo di delimitazione definito dal workspace.
        Parametri:
            workspace (tuple): Una tupla di quattro valori (xmin, ymin, xmax, ymax) 
                               che rappresentano il rettangolo di delimitazione del workspace.
        Returns:
            bool: True se tutti i punti sono all'interno del workspace, False altrimenti.
        """
        # footprint circolare
        if (workspace[0] + self.footprint['radius'] <= self.x <= workspace[2] - self.footprint['radius'] and
            workspace[1] + self.footprint['radius'] <= self.y <= workspace[3] - self.footprint['radius']):
            return True
        return False
        
    def lidar(self, Ship: ShipObstacle) -> Tuple[np.ndarray, set[int]]:
        """
        Simula un sensore LiDAR per rilevare un ostacolo definito da un oggetto ShipObstacle.
        Args:
            Ship (ShipObstacle): L'ostacolo da rilevare.
            lidar_params (dict): Parametri del LiDAR, inclusi 'max_range' (portata massima), 
                                 'n_beams' (numero di raggi) e 'FoV' (campo visivo).
        Returns:
            np.ndarray: Distanze rilevate per ciascun raggio.
        """
        pose = self.get_state()[:3]
        # Simulazione del sensore LiDAR
        ranges, _, seen_segments = lidar(pose, Ship, self.lidar_params)
        
        return ranges, seen_segments

if __name__ == '__main__':

    """Test della classe ROV"""
    # # Parametri del LiDAR
    # lidar_params = {
    #     'max_range': 2,
    #     'n_beams': 20,
    #     'FoV': np.pi
    # }
    # footprint = {'radius': 0.5}

    # # Posizione del sensore
    # pose = np.array([5.1, 5.0, 0])

    # # Creazione di un ostacolo
    # Ship = ShipObstacle((4,4), inflation_radius=footprint['radius'])

    # # Simulazione del sensore LiDAR
    # # ranges, angles, seen_segments = lidar(pose, Ship, lidar_params)
    
    # # print(ranges)
    # # print(angles)
    # # print(seen_segments)

    # # Creazione di un ROV
    # rov = Rov(init_pose=pose, footprint=footprint, lidar_parms=lidar_params)
    # ranges, seen_segments = rov.lidar(Ship)
    # print(ranges)
    # print(seen_segments)
    # collision = rov.collision_check(Ship)
    # print(collision)

    """Test wrapToPi"""

    # agent = Rov()

    # for i in range(10000):
    #     angle = np.random.uniform(0,1000)
    #     wrapped_angle = agent.wrapToPi(angle)
    #     if wrapped_angle < -np.pi or wrapped_angle >= np.pi:
    #         print(f"Errore: {angle} -> {wrapped_angle}")
    #         break
    
    # print("Test superato!")

    """ Test saturate """
    agent = Rov()
    done = False
    for i in range(10):
        dvx = - 0.5
        if done:
            dvx = 0.0
        dvy = 0.0
        domega = 0.0
        saturation = agent.zeno_kinematics(dvx, dvy, domega)
        if saturation[0]:
            done = True 
        print(saturation)
        print(agent.get_zeno_state()[3::])

