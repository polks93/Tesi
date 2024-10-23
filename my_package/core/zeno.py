import numpy as np
import warnings
from typing import Tuple, Sequence, Optional

from obstacle_simulation import ShipObstacle, lidar


class Zeno:
    def __init__(
            self, 
            init_pose: np.ndarray = np.array([0.0, 0.0, 0.0]), 
            footprint: dict ={}, 
            lidar_parms: dict ={'FoV': np.deg2rad(60), 'max_range': 3.0, 'n_beams': 10}
            ) -> None:

        # Inizializzo lo stato del ROV
        self.init_pose = init_pose
        self.x = self.init_pose[0]
        self.y = self.init_pose[1]
        self.theta = self.init_pose[2]
        self.v_surge = 0.0
        self.v_sway = 0.0
        self.omega = 0.0
        
        # Definise le velocità massime (modello di Zeno)
        self.max_v = 0.2
        self.max_omega = 0.15

        # Definisco i parametri del controllore
        self.K = {'yaw': 20.0, 'omega': 50.0, 'v_surge': 0.25, 'v_sway': 0.25}	

        # Definisce la footprint dell'uniciclo
        if 'radius' not in footprint:
            warnings.warn("footprint non definita correttamente. Imposto la footprint circolare con raggio 0.5.")
            footprint = {'radius': 0.5}
        self.footprint = footprint

        # Definisce i parametri del sensore LIDAR
        self.lidar_params = lidar_parms

    def reset(self, init_pose: Optional[np.ndarray] = None) -> None:

        if init_pose is not None:
            self.init_pose = init_pose
        self.x = self.init_pose[0]
        self.y = self.init_pose[1]
        self.theta = self.init_pose[2]
        self.v_surge = 0.0
        self.v_sway = 0.0

    def saturate(self, value: float, max_value: float) -> float:
        """
        Limita il valore assoluto di un numero al massimo specificato.
        Args:
            value (float): Il valore da limitare.
            max_value (float): Il valore massimo assoluto consentito.
        Returns:
            float: Il valore limitato
        """
        if value > max_value:
            return max_value
        elif value < - max_value:
            return - max_value
        return value 

    def wrapToPi(self, angle: float):
        """
        Converte l'angolo di input nell'intervallo [-pi, pi].
        Args:
            angle (float): L'angolo in radianti da convertire.
        Returns:
            float: L'angolo convertito nell'intervallo [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi     
    
    def controller(
            self, 
            v_surge_des: float  = 0.0, # Valore assoluto [m/s]
            v_sway_des: float   = 0.0, # Valore assoluto [m/s]
            yaw_rel: float      = 0.0, # Angolo relativo [rad]
            dt: float           = 0.1
            ) -> None:
        """
        Controlla e aggiorna le velocità del sistema basato sugli input desiderati.
        Args:
            v_surge_des (float): Velocità desiderata lungo l'asse longitudinale (surge).
            v_sway_des (float): Velocità desiderata lungo l'asse trasversale (sway).
            yaw_des (float): Orientamento desiderato (yaw).
            dt (float): Time step in secondi.
        Returns:
            None
        """  

        # Saturo gli ingressi per rispettare i limiti di velocità
        v_surge_des = self.saturate(v_surge_des, self.max_v)
        v_sway_des = self.saturate(v_sway_des, self.max_v)

        # Calcolo degli errori
        yaw_error       = self.wrapToPi(yaw_rel)
        v_surge_error   = v_surge_des - self.v_surge
        v_sway_error    = v_sway_des - self.v_sway

        # Dinamica del secondo ordine per alpha
        alpha = self.K['yaw'] * yaw_error - self.K['omega'] * self.omega

        # Dinamica del primo ordine per a_surge e a_sway
        a_surge = self.K['v_surge'] * v_surge_error
        a_sway = self.K['v_sway'] * v_sway_error

        # Applico gli input di controllo
        self.omega      = self.saturate(self.omega + alpha * dt / 100, self.max_omega)
        self.v_surge    = self.saturate(self.v_surge + a_surge * dt, self.max_v)
        self.v_sway     = self.saturate(self.v_sway + a_sway * dt, self.max_v)

        return None
    
    def kinematics(self, dt: float):
        """
        Aggiorna la posizione e l'orientamento di zeno in base alla velocità e al tempo trascorso.
        Args:
            dt (float): L'intervallo di tempo durante il quale aggiornare la cinematica.
        Effettua i seguenti aggiornamenti:
        - Aggiorna l'angolo theta di Zeno utilizzando la funzione wrapToPi per mantenere l'angolo entro i limiti [-pi, pi].
        - Aggiorna la posizione x di Zeno in base alla velocità di avanzamento (v_surge) e alla velocità laterale (v_sway).
        - Aggiorna la posizione y di Zeno in base alla velocità di avanzamento (v_surge) e alla velocità laterale (v_sway).
        """

        self.theta = self.wrapToPi(self.theta + self.omega * dt)
        self.x += (self.v_surge * np.cos(self.theta)  - self.v_sway * np.sin(self.theta)) * dt
        self.y += (self.v_surge * np.sin(self.theta)  + self.v_sway * np.cos(self.theta)) * dt

    def get_state(self):
        """
        Restituisce lo stato attuale del sistema come array numpy.
        Returns:
            np.ndarray: Un array contenente i seguenti valori:
                - self.x: La posizione x.
                - self.y: La posizione y.
                - self.theta: L'angolo theta.
                - self.v_surge: La velocità di avanzamento.
                - self.v_sway: La velocità di deriva.
                - self.omega: La velocità angolare.
        """
        return np.array([self.x, self.y, self.theta, self.v_surge, self.v_sway, self.omega])
    
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
    zeno = Zeno(init_pose=np.array([0.0, 0.0, 0.0]), footprint={'radius': 0.5})
    done = False
    surge_des = 0.25
    sway_des = -0.25
    theta_des = np.deg2rad(20)
    theta = zeno.get_state()[2]
    theta_his = [np.rad2deg(theta)]
    surge_his = [0.0]
    omega_his = [0.0]
    sway_his = [0.0]
    time = [0.0]
    dt = 0.001
    T = 0.0
    while T < 50.0:
        theta = zeno.get_state()[2]
        theta_rel = theta_des - theta
        zeno.controller(surge_des, sway_des, theta_rel, dt)
        zeno.kinematics(dt)
        state = zeno.get_state()
        surge = state[3]
        theta = np.rad2deg(state[2])
        sway = state[4]
        omega = state[5]

        surge_his.append(surge)
        theta_his.append(theta)
        omega_his.append(omega)
        sway_his.append(sway)
        time.append(T)
        T = T + dt
    import matplotlib.pyplot as plt
    # time = list(range(len(surge_his)))

    # Creazione della figura con 4 subplot (2x2)
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    # Primo subplot: Surge
    axs[0].plot(time, surge_his, label="Surge")
    axs[0].set_title('Surge')
    axs[0].set_ylabel('Values')
    axs[0].grid(True)

    # Secondo subplot: Omega
    axs[1].plot(time, omega_his, label="Omega", color='orange')
    axs[1].set_title('Omega')
    axs[1].set_ylabel('Values')
    axs[1].grid(True)

    # Terzo subplot: Theta
    axs[2].plot(time, theta_his, label="Theta", color='green')
    axs[2].set_title('Theta')
    axs[2].set_ylabel('Values')
    axs[2].grid(True)

    # Quarto subplot: Sway
    axs[3].plot(time, sway_his, label="Sway", color='red')
    axs[3].set_title('Sway')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Values')
    axs[3].grid(True)

    # Aggiunta di spazi tra i subplot per una migliore visualizzazione
    plt.tight_layout()

    # Mostrare il plot
    plt.show()