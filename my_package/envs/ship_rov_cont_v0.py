import pygame
import numpy        as np
import gymnasium    as gym

from gymnasium  import spaces
from typing     import Tuple, Dict, Any, Optional, Union, Set

from obstacle_simulation import ShipObstacle

from my_package.core.zeno    import Zeno

class ShipRovContEnv(gym.Env):
    """Upgrade di ShipUniCont-v1
    Aggiunta azioni per v_sway
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self,
            Options: Dict[str, Any],
            workspace: Tuple                = (0, 0, 8, 8), 
            render_mode: Optional[str]      = None
            ) -> None:
            """
            Inizializza un'istanza della classe con i parametri specificati.
            Args:
                workspace (Tuple, optional): Le coordinate dello spazio di lavoro sotto forma di tuple (x_min, y_min, x_max, y_max). Default è (0, 0, 10, 10).
                ship (list, optional): Le coordinate della nave sotto forma di tuple (x_min, y_min, x_max, y_max). Default è (4, 4, 6, 6).
                init_pose (Sequnece, optional): La posizione iniziale dell'agente sotto forma di tuple (x, y, theta). Default è (1, 1, np.pi/4).
                render_mode (Optional[str], optional): La modalità di rendering. Default è None.
            Returns:
                None
            """

            # Import workspace
            self.workspace = workspace
            self.min_x = workspace[0]
            self.min_y = workspace[1]
            self.max_x = workspace[2]
            self.max_y = workspace[3]
            self.ws_center = ((self.max_x - self.min_x) / 2, (self.max_y - self.min_y) / 2)

            # Inizializzo agente
            if Options['init_pose'] is None:
                self.init_pose = self.get_random_init_pose()
                self.random_init_pose = True
            else:
                self.init_pose = np.array(Options['init_pose'])
                self.random_init_pose = False

            # Parametri agente
            self.dt                     = 1.0 / self.metadata["render_fps"]
            self.frame_per_step         = Options['frame_per_step']
            self.agent_radius           = Options['agent_radius']
            self.frontal_safe_distance  = Options['frontal_safe_distance']
            self.v_surge_max            = Options['v_surge_max']
            self.v_sway_max             = Options['v_sway_max']
            self.yaw_rel_max            = Options['yaw_rel_max']

            # Parametri del LiDAR
            self.lidar_params   = Options['lidar_params']
            self.draw_lidar     = Options['draw_lidar']
            self.lidar_angles = np.linspace(- self.lidar_params['FoV'] / 2, self.lidar_params['FoV'] / 2, self.lidar_params['n_beams'])

            # Creo un agente del tipo Unicylce
            self.agent = Zeno(init_pose=self.init_pose, footprint={'radius': self.agent_radius}, lidar_parms=self.lidar_params)
            
            # Definisco il nuemero di azioni disponibili e dimensioni dello stato
            self.n_actions = 3

            # Definisco lo spazio delle azioni
            self.action_low     = - np.ones(self.n_actions)
            self.action_high    =   np.ones(self.n_actions)
            self.action_space   = spaces.Box(low=self.action_low, high=self.action_high, shape=(self.n_actions,), dtype=np.float64)

            # Definisco lo spazio delle osservazioni
            self.state_dim = 6
            self.num_observation = self.state_dim + Options['lidar_params']['n_beams']
            low     = np.zeros(self.num_observation).astype(np.float64)
            high    = np.ones(self.num_observation).astype(np.float64)
            self.observation_space = spaces.Box(low=low, high=high, shape=(self.num_observation,), dtype=np.float64)

            # Creo un ostacolo del tipo ShipObstacle
            ship_scale_factor = Options['ship_scale_factor']
            self.Ship = ShipObstacle(self.ws_center, inflation_radius=self.agent_radius, scale=ship_scale_factor)

            # Import dati ostacolo
            self.random_ship        	    = Options['generate_random_ship']
            self.workspace_safe_distance    = Options['workspace_safe_distance']

            # Inizializzo rendering
            self.render_mode = render_mode
            self.screen = None
            self.clock  = None
            self.screen_size = (800, 800)
            self.scale = self.screen_size[0] / (self.max_x - self.min_x)

            self.max_steps = Options['max_steps']

    def get_random_init_pose(self) -> np.ndarray:
        """
        Genera un agente in una posizione casuale all'interno dei confini definiti.
        La funzione calcola i possibili punti di spawn lungo gli assi x e y, 
        scegliendo casualmente tra questi punti e i bordi del confine. 
        Successivamente, calcola l'angolo theta tra il punto di spawn e il centro del confine,
        in modo che l'agente sia orientato verso il centro del confine.
        Returns:
            Tuple: Una tupla contenente le coordinate x, y e l'angolo theta del punto di spawn.
        """
        distance_from_boundary = 1.0
        possible_spawn_points_x = np.linspace(self.min_x + distance_from_boundary, self.max_x - distance_from_boundary, 100)
        possible_spawn_points_y = np.linspace(self.min_y + distance_from_boundary, self.max_y - distance_from_boundary, 100)

        randomize_x = np.random.choice([True, False])

        # y fisso (sopra o sotto) e x random
        if randomize_x:
            x = np.random.choice(possible_spawn_points_x)
            y = np.random.choice([self.min_y + distance_from_boundary, self.max_y - distance_from_boundary])

        # x fisso (a sinistra o a destra) e y random
        else:
            x = np.random.choice([self.min_x + distance_from_boundary, self.max_x - distance_from_boundary])
            y = np.random.choice(possible_spawn_points_y)
        
        # Punto centrale del workspace
        Cx = (self.max_x - self.min_x) / 2
        Cy = (self.max_y - self.min_y) / 2
        
        # Calcolo l'angolo theta tra il punto di spawn e il centro del workspace
        theta = np.arctan2(Cy - y, Cx - x)
        # Aggiungo deviazione random all'angolo theta
        d_theta = np.random.uniform(-np.pi/6, np.pi/6)
        theta = theta + d_theta

        return np.array([x, y, theta])
    
    def get_obs(self) -> Tuple[np.ndarray, Set[int]]:
        """
        Restituisce l'osservazione corrente dell'ambiente.

        Returns:
            observation (np.ndarray): L'osservazione corrente dell'ambiente strutturata in questo modo:
            - x:                    coordinata x dell'agente
            - y:                    coordinata y dell'agente
            - theta:                orientamento dell'agente
            - v_surge:              velocità di avanzamento dell'agente (solo per modello AUV)
            - v_sway:               velocità laterale dell'agente (solo per modello AUV)
            - omega:                velocità angolare dell'agente
            - ranges(n_beams):      distanza misurata dal LiDAR per ogni raggio
        """
        # Init vettore di osservazione
        observation = np.zeros(self.num_observation).astype(np.float64)

        # Salvo le prime 3 componenti dello stato dell'agente e le normalizzo
        state = self.agent.get_state()
        norm_x      = ( state[0] - self.min_x ) / ( self.max_x - self.min_x )
        norm_y      = ( state[1] - self.min_y ) / ( self.max_y - self.min_y )
        norm_theta  = ( state[2] + np.pi ) / ( 2*np.pi )
       
        norm_v_surge    = ( 1 + state[3] / self.v_surge_max ) / 2
        norm_v_sway     = ( 1 + state[4] / self.v_sway_max ) / 2
        norm_omega      = ( 1 + state[5] / self.agent.max_omega ) / 2

        # Ottengo i dati del lidar e li normalizzo
        ranges, seen_segments_id = self.agent.lidar(self.Ship)
        norm_ranges = ranges / self.lidar_params['max_range']

        # Aggiorno vettore di osservazione
        observation[0]      = norm_x                    # Coordinata x
        observation[1]      = norm_y                    # Coordinata y
        observation[2]      = norm_theta                # Orientamento
        observation[3]      = norm_v_surge              # Velocità di avanzamento
        observation[4]      = norm_v_sway               # Velocità laterale
        observation[5]      = norm_omega                # Velocità angolare
        observation[6:]     = norm_ranges               # Dati LiDAR

        return observation, seen_segments_id
    
    def get_info(self) -> Dict[str, Any]:
        """
        Restituisce le informazioni aggiuntive sull'ambiente.

        Returns:
            info (dict): Informazioni aggiuntive sull'ambiente.
        """
        # Calcolo quanti segmenti sono stati visti dall'agente
        seen_segments = sum(1 for segment in self.segments.values() if segment.seen)
        coverage = round(seen_segments / self.n_segments * 100, 2)

        # info = self.status | {'coverage' : coverage}
        info = {**self.status, 'coverage': coverage}
        return info   
    
    def reset(self, *, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None)-> Tuple[np.ndarray, Dict[str, Any]]:
        
        """
        Reimposta l'ambiente a uno stato iniziale, restituendo un'osservazione iniziale e informazioni.

        Questo metodo genera un nuovo stato iniziale con una certa casualità per garantire che l'agente esplori lo spazio degli stati.
        La casualità può essere controllata con il parametro seed. Se l'ambiente ha già un PRNG e seed=None,
        l'RNG non viene reimpostato.

        Args:
            seed (int opzionale): Seed per inizializzare il PRNG dell'ambiente. Se None, un seed viene scelto dall'entropia.
            options (dict opzionale): Informazioni aggiuntive per specificare come reimpostare l'ambiente.

        Returns:
            observation (np.ndarray): Osservazione dello stato iniziale, un elemento di observation_space.
            info (dict): Informazioni ausiliarie che completano l'osservazione.
        """
        super().reset(seed=seed, options=options)
        self.step_count = 0
        self.Ship.reset_ship()

        if self.random_ship:
            self.Ship.random_placement(self.workspace, self.workspace_safe_distance)

        # Dizionario contenente i segmenti che dividono l'ostacolo
        self.segments = self.Ship.copy_segments()
        self.n_segments = len(self.segments)
        self.seen_segments = 0
        # Reimposto l'agente su una posizione casuale in modo che non veda
        # nessun segmento dell'ostacolo all'inizio dell'episodio
        if self.random_init_pose:
            done = False
            while not done:  
                init_pose = self.get_random_init_pose()

                self.agent.reset(init_pose=init_pose)
                _, seen_segments = self.agent.lidar(self.Ship)

                if len(seen_segments) == 0:
                    done = True
        else:
            init_pose = self.init_pose
            self.agent.reset(init_pose=init_pose)
            _, seen_segments = self.agent.lidar(self.Ship)
            if len(seen_segments) > 0:
                raise UserWarning(f"Init pose defined too close to the ship. {len(seen_segments)} segments are visible.")

        self.status = {'goal': False, 'collision': False, 'out_of_bounds': False, 'time_limit': False}   

        observation, seen_segments  = self.get_obs()
        info                        = self.get_info()

        if len(seen_segments) > 0:
            raise UserWarning(f"Agent spawned too close to the ship. {len(seen_segments)} segments are visible.")

        if self.render_mode == "human":
            self.render()

        return observation, info   
    
    def action_to_controls(self, action: np.ndarray) -> Dict:
        """
        Converte l'azione scelta dall'agente nei controlli necessari per l'agente.
        L'azione relativa allo yaw è un valore tra -1 e 1 che viene prima convertito in -pi/4 e pi/4. 
        In questo modo il valore corrisponde all'angolo relativo desiderato.
        Args:
            action (np.ndarray): L'azione scelta dall'agente.
        Returns:
            dict: Un dizionario contenente i controlli necessari per l'agente.
        """

        v_surge = action[0] * self.v_surge_max
        v_sway  = action[1] * self.v_sway_max
        yaw_rel = action[2] * self.yaw_rel_max

        controls = {'v_surge': float(v_surge), 'v_sway': float(v_sway), 'yaw_rel': float(yaw_rel)}
        return controls
        
    def segments_check(self, seen_segments_id) -> Tuple[int, bool]:
        """
        Verifica i segmenti visti e aggiorna il loro stato.
        Args:
            seen_segments_id (list): Lista degli ID dei segmenti visti nell'ultima osservazione
        Returns:
            Tuple[int, bool]: Una tupla contenente il numero di nuovi segmenti visti e un booleano che indica se 
            almeno un segmento è stato visto.
        """

        new_segnments = 0
        any_segment_seen = False

        if len(seen_segments_id) == 0:
            return new_segnments, any_segment_seen

        any_segment_seen = True

        for id in seen_segments_id:
            if self.segments[id].seen == False:
                self.segments[id].seen = True
                new_segnments += 1

        return new_segnments, any_segment_seen
    
    def goal_check(self) -> bool:
        """
        Verifica se l'agente ha visto tutti i segmenti.

        Returns:
            bool: True se l'agente ha visto tutti i segmenti, False altrimenti.
        """
        return all([segment.seen for segment in self.segments.values()])
    
    def get_reward(self, observation: np.ndarray, seen_segments_id: Set[int]) -> Tuple[float, bool]:
        """
        Calcola il reward per l'agente basato sull'osservazione corrente e verifica se l'episodio è terminato.
        Args:
            observation (list): Lista contenente i dati dell'osservazione corrente.
        Returns:
            Tuple[float, bool, bool]: Una tupla contenente:
                - reward (float): Il reward calcolato per l'osservazione corrente.
                - terminated (bool): Indica se l'episodio è terminato.
        """

        reward = 0.0
        terminated = False

        # Reward possibili
        r_step          = - 0.01
        r_v_negative    = - 0.01
        r_new_segment   = 1.0
        r_collision     = - 25.0
        r_out_of_bounds = - 25.0
        r_goal          = 100.0

        # Reward negativa per ogni step
        reward += r_step

        # Piccola reward negativa per velocità di surge negativa
        v_surge = self.agent.get_state()[3]
        if v_surge < 0:
            reward += r_v_negative * abs(v_surge)

        # Calcolo il numero di nuovi segmenti visti e se almeno un segmento è stato visto
        new_segments_seen, _ = self.segments_check(seen_segments_id)
        # Reward per i nuovi segmenti visti
        reward += new_segments_seen * r_new_segment

        # Check eventi che terminano l'episodio
        # Max steps
        if self.step_count >= self.max_steps:
            self.status['time_limit'] = True
            terminated = True
            return reward, terminated

        # Out of bounds
        if not self.agent.boundary_check(self.workspace):
            reward += r_out_of_bounds
            self.status['out_of_bounds'] = True
            terminated = True   
            return reward, terminated

        # Goal raggiunto
        if self.goal_check():
            reward += r_goal
            self.status['goal'] = True
            terminated = True
            return reward, terminated
        
        # Collisione
        if self.agent.collision_check(self.Ship):
            reward += r_collision
            self.status['collision'] = True
            terminated = True
            return reward, terminated

        return reward, terminated    
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Esegue un passo nell'ambiente in base all'azione fornita.

        Questo metodo aggiorna lo stato del robot in base all'azione scelta, 
        calcola la nuova osservazione, la ricompensa, e verifica se l'episodio è terminato.
        Se la modalità di rendering è impostata su 'human', viene renderizzato un frame.

        Args:
            action np.float64: L'azione scelta dall'agente.

        Returns:
            observation (np.array): La nuova osservazione dell'ambiente.
            reward (float): La ricompensa ottenuta dopo aver eseguito l'azione.
            terminated (bool): Indica se l'episodio è terminato.
            truncated (bool): Indica se l'episodio è stato troncato.
            info (dict): Informazioni aggiuntive sull'ambiente.
        """
        self.step_count += 1
        controls = self.action_to_controls(action)

        new_segments_id = set()

        for _ in range(self.frame_per_step):
            # Eseguo l'azione 
            self.agent.controller(v_surge_des=controls['v_surge'], v_sway_des=controls['v_sway'], yaw_rel=controls['yaw_rel'], dt=self.dt)
            self.agent.kinematics(self.dt)
            
            # Calcolo la nuova osservazione
            observation, step_seen_segments_id = self.get_obs()

            # Aggiungo i nuovi segmenti
            for id in step_seen_segments_id:
                new_segments_id.add(id)
            
        # Calcolo le reward e verifico se l'episodio è terminato
        reward, terminated = self.get_reward(observation, new_segments_id)

        truncated = False
        info = self.get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info   

    def draw_ship(self) -> None:
        """
        Disegna le navi sullo schermo.
        Questo metodo scorre attraverso la lista delle navi e disegna ciascuna nave
        come un rettangolo sulla superficie di Pygame. La posizione e le dimensioni
        del rettangolo sono calcolate in base alle coordinate della nave e alla scala
        della mappa.
        """
        # Disegno la nave
        ship_points = self.Ship.points
        pixel_ship_points = []
        for point in ship_points:
            x = round((point[0] - self.min_x) * self.scale)
            y = self.screen_size[1] - round((point[1] - self.min_y) * self.scale)
            pixel_ship_points.append((x, y))

        pygame.draw.polygon(self.surf, (0, 0, 0), pixel_ship_points, 0)

        # Disegno i segmenti che compongono l'ostacolo
        for i in range(len(self.segments)):
            x1 = round((self.segments[i].start_point[0] - self.min_x) * self.scale)
            y1 = self.screen_size[1] - round((self.segments[i].start_point[1] - self.min_y) * self.scale)
            x2 = round((self.segments[i].end_point[0] - self.min_x) * self.scale)
            y2 = self.screen_size[1] - round((self.segments[i].end_point[1] - self.min_y) * self.scale)
            if self.segments[i].seen:
                pygame.draw.line(self.surf, (0, 255, 0), (x1, y1), (x2, y2), 4)
            else:
                pygame.draw.line(self.surf, (255, 0, 0), (x1, y1), (x2, y2), 4)

    def draw_agent(self) -> None:
        """
        Disegna l'agente e il suo campo visivo sulla superficie di rendering.
        Questo metodo esegue le seguenti operazioni:
        1. Calcola la posizione dell'agente in pixel e il raggio dell'agente in pixel.
        2. Disegna un cerchio che rappresenta l'agente.
        3. Disegna il campo visivo totale dell'agente.
        4. Se `self.draw_lidar` è True, disegna le linee che rappresentano gli angoli del lidar.
        """
        
        agent_pixel_position = (
            round((self.agent.x - self.min_x) * self.scale),
            round(self.screen_size[1] - (self.agent.y - self.min_y) * self.scale)
        )
        agent_radius_pixel = round(self.agent_radius * self.scale)

        # Disegno agente
        pygame.draw.circle(self.surf, (0, 0, 255), agent_pixel_position, agent_radius_pixel)

        # Disengo il FoV dell'agente
        range_pixel = round(self.lidar_params['max_range'] * self.scale)
        FoV = self.lidar_params['FoV']
        x1 = agent_pixel_position[0] + round(range_pixel * np.cos(self.agent.theta - FoV/2))
        y1 = agent_pixel_position[1] - round(range_pixel * np.sin(self.agent.theta - FoV/2))
        x2 = agent_pixel_position[0] + round(range_pixel * np.cos(self.agent.theta + FoV/2))
        y2 = agent_pixel_position[1] - round(range_pixel * np.sin(self.agent.theta + FoV/2))

        pygame.draw.line(self.surf, (255, 0, 0), agent_pixel_position, (x1, y1), 1)
        pygame.draw.line(self.surf, (255, 0, 0), agent_pixel_position, (x2, y2), 1)
        pygame.draw.arc(
            self.surf, 
            (255, 0, 0), 
            (agent_pixel_position[0] - range_pixel, agent_pixel_position[1] - range_pixel, 2*range_pixel, 2*range_pixel), 
            self.agent.theta - FoV/2,
            self.agent.theta + FoV/2,   
            1
        )

        # Visualizza i raggi del LiDAR
        if self.draw_lidar:
            for i, angle in enumerate(self.lidar_angles):
                x = agent_pixel_position[0] + round(range_pixel * np.cos(self.agent.theta + angle))
                y = agent_pixel_position[1] - round(range_pixel * np.sin(self.agent.theta + angle))
                pygame.draw.line(self.surf, (255, 0, 0), agent_pixel_position, (x, y), 1)

    def render(self):
        """
        Calcola i frame di rendering come specificato da render_mode durante l'inizializzazione dell'ambiente.

        I metadati dell'ambiente (env.metadata["render_modes"]) dovrebbero contenere i modi possibili per implementare i render modes.

        Nota:
        -  None (default): nessun render viene calcolato.
        - "human":  L'ambiente viene continuamente renderizzato nel display corrente o terminale per consumo umano. 
                    Questo rendering dovrebbe avvenire durante step e render non ha bisogno di essere chiamato. Restituisce None.
        - "rgb_array":  Restituisce un singolo frame che rappresenta lo stato corrente dell'ambiente.
                        Un frame è un np.ndarray con forma (x, y, 3) che rappresenta i valori RGB per un'immagine di x per y pixel.
        
        Assicurati che i metadati della tua classe includano la chiave "render_modes" con la lista dei modi supportati.
        """
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(self.screen_size)
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface(self.screen_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()


        self.surf = pygame.Surface(self.screen_size)
        self.surf.fill((255, 255, 255))

        # Disegna su self.surf
        self.draw_ship()
        self.draw_agent()
        # self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        # Gestione rgb_array
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    Options = {
        'generate_random_ship':             True,
        'ship_perimeter':                   12,
        'workspace_safe_distance':          2,
        'ship_scale_factor':                0.9,
        'segments_lenght':                  0.25,
        'frame_per_step':                   10,
        'init_pose':                        None,
        'agent_radius':                     0.1,
        'v_surge_max':                      0.2,
        'v_sway_max':                       0.2,	
        'yaw_rel_max':                      np.pi/4,
        'frontal_safe_distance':            0.5,
        'lidar_params':                     {'n_beams': 10, 'max_range': 2.0, 'FoV': np.pi/2},
        'draw_lidar':                       True,
        'max_steps':                        2000
    }	

    # env = ShipRovContEnv(render_mode='human', Options=Options, workspace=(0,0,8,8))
    env = gym.make("ShipRovCont-v0", render_mode="human", Options=Options, workspace=(0, 0, 18, 18))
    low = env.action_low
    high = env.action_high
    print(f"Action space: {low} {high}")

    observation, info = env.reset()
    print(observation)
    # Esegui 100 passi casuali
    for _ in range(1000):
        action = env.action_space.sample()  # Esegui un'azione casuale
        # action[0] = 0.5
        # action[2] = 0.0

        observation, reward, terminated, truncated, info = env.step(action)
        if info['collision']:
            print("Collisione")
            print(info)
        # v_surge = env.agent.get_state()[3]
        # print(v_surge)
        # print(action)
        # print(observation)
        if terminated or truncated:
            observation, info = env.reset()

    # for i in range(100):
    #     state = env.agent.get_zeno_state()
    #     action = 1
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(observation)
    #     if terminated or truncated:
    #         observation, info = env.reset()
    
    env.close()