import gymnasium    as gym
from gymnasium.core import RenderFrame
import numpy        as np
import pygame
from typing import Any, Dict, Optional, Tuple, Union
from gymnasium  import spaces
from my_package import Unicycle, is_in_FoV, is_in_rect
from numpy.typing import NDArray

class RewardExplorerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30} 

    def __init__(
            self, 
            workspace: Tuple = (0, 0, 10, 10), 
            render_mode: Optional[str] = None
            ) -> None:
        """
        Inizializza l'ambiente di esplorazione della nave.

        Args:
            workspace (tuple): Limiti del workspace definiti come (min_x, min_y, max_x, max_y).s
            render_mode (str, opzionale): Modalità di rendering, può essere 'human' o 'rgb_array'.

        Crea un'istanza della classe Unicycle, definisce gli spazi delle osservazioni e delle azioni,
        imposta la modalità di rendering e inizializza i dati per il rendering e la gridmap.
        """
        super(RewardExplorerEnv, self).__init__()

        # Init workspace
        self.workspace = workspace
        self.min_x = workspace[0]
        self.min_y = workspace[1]
        self.max_x = workspace[2]
        self.max_y = workspace[3]
        self.reward_radius = 0.05
        self.n_rewards = 10

        # Creo un'istanza della classe Unicycle
        self.init_pose = [1, 1, 0]
        self.agent = Unicycle(init_pose=self.init_pose, footprint={'radius': 0.1})
        self.robot_radius = self.agent.footprint['radius']
        self.dt = 1.0 / self.metadata['render_fps']

        # Definizione campo visivo e raggio del sensore di prossimità
        self.FoV = np.pi/3
        self.range = 1.0

        # Definizione zona target
        self.target_area = (self.max_x - 2, self.max_y - 2, self.max_x - 1, self.max_y - 1)

        # Definizione spazio delle osservazioni e delle azioni
        self.action_space = spaces.Discrete(3)
        self.action_to_controls = {
            0: 0.0, 
            1: self.agent.max_omega, 
            2: -self.agent.max_omega,
        }

        # Definizione spazio delle osservazioni
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Imposto la render mode, se passata correttamente altrimenti passo un errore
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Dati rendering
        self.window_size = (800, 800)
        self.scale = self.window_size[1] / (self.max_y - self.min_y)
        self.window = None
        self.clock = None

    def get_obs(self) -> np.ndarray:
        """
        Ottiene lo stato attuale del robot e lo normalizza per restituire un'osservazione.

        Il metodo ottiene le coordinate x, y e l'orientamento theta del robot. 
        Questi valori vengono poi normalizzati per essere compresi nell'intervallo [-1, 1].
        La normalizzazione di x e y viene effettuata rispetto ai limiti del workspace, 
        mentre theta viene normalizzato rispetto a [-pi, pi].

        Returns:
            observation (np.array): Un array contenente le coordinate normalizzate x, y e theta del robot.
        """
        # Ottieni lo stato del robot
        x, y, theta = self.agent.get_state()[:3] 
        # Normalizzazione di x e y dall'intervallo [min_x, max_x] a [-1, 1]
        x_norm = 2 * (x - self.min_x) / (self.max_x - self.min_x) - 1
        y_norm = 2 * (y - self.min_y) / (self.max_y - self.min_y) - 1

        # Normalizzazione di theta dall'intervallo [-pi, pi] a [-1, 1]
        theta_norm = theta / np.pi

        # Numero di ricompense raccolte
        rewards_collected = self.status['reward_collected']

        # Normalizzazione delle ricompense raccolte nell'intervallo [-1, 1]
        max_possible_rewards = self.n_rewards 
        norm_rewards_collected = (rewards_collected / max_possible_rewards) * 2 - 1

        # Combina le osservazioni normalizzate
        observation = np.array([x_norm, y_norm, theta_norm, norm_rewards_collected], dtype=np.float32)
        return observation

    def get_info(self) -> Dict:
        """
        Restituisce informazioni aggiuntive sull'ambiente.

        Questo metodo restituisce informazioni aggiuntive sull'ambiente, 
        come il workspace, la posizione del robot e i limiti del workspace.

        Returns:
            info (dict): Un dizionario contenente informazioni aggiuntive sull'ambiente.
        """
        info = {
            "workspace": self.workspace,
            "robot_position": self.agent.get_state()[:2],
            "workspace_limits": (self.min_x, self.min_y, self.max_x, self.max_y)
        }	
        return self.status | info
    
    def spawn_rewards(self, seed: int = 10) -> Dict[int, Dict[str, Any]]:
        """
        Genera un dizionario di ricompense casuali.
        Parametri:
        - num_rewards (int): Il numero di ricompense da generare (default: 10).
        - seed (int): Il seme per la generazione casuale (default: 10).
        Ritorna:
        - reward_dict (dict): Un dizionario contenente le ricompense generate. Ogni ricompensa è rappresentata da un indice
                              e ha le seguenti chiavi:
                              - 'position': La posizione della ricompensa, rappresentata da una lista di due coordinate (x, y).
                              - 'collected': Un flag booleano che indica se la ricompensa è stata raccolta o meno.
        """
        
        reward_dict = {}
        np.random.seed(seed)

        # Reward generate ad almeno 0.1 unità dai bordi
        min_x = self.min_x + 0.1
        max_x = self.max_x - 0.1
        min_y = self.min_y + 0.1
        max_y = self.max_y - 0.1

        # Popolo il dizionario delle ricompense
        for i in range(self.n_rewards):
            # Genero posizione random
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            # Verifico che la posizione non sia all'interno della target area
            while is_in_rect((x, y), self.target_area) or is_in_FoV(self.agent.get_state()[:3], self.FoV, self.range, [x,y]):
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
            # Aggiungo la ricompensa al dizionario
            reward_dict[i] = {'position': [x,y], 'collected': False}
            
        return reward_dict
    
    def reset(self, *, 
              seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resetta l'ambiente all'inizio di un nuovo episodio.

        Questo metodo resetta lo stato del robot e restituisce l'osservazione iniziale.
        Se la modalità di rendering è impostata su 'human', viene renderizzato il primo frame.

        Args:
            seed (int, opzionale): Il seme per il generatore di numeri casuali.
            options (dict, opzionale): Opzioni aggiuntive per il reset.

        Returns:
            observation (np.array): L'osservazione iniziale dell'ambiente.
            info (dict): Informazioni aggiuntive sull'ambiente.
        """
        super().reset(seed=seed, options=options)
        self.agent.reset()
        self.reward_dict = self.spawn_rewards()
        # Inizializzo lo status agente
        self.status = {'goal': False, 'reward_collected': 0, 'collision': False}
        observation = self.get_obs()
        info = self.get_info()

        if self.render_mode == 'human':
            self._render_frame()
        return observation, info
    
    def reward_check(self) -> int:
        reward_collected = 0
        for i in range(self.n_rewards):
            if not self.reward_dict[i]['collected']:
                reward_pos = self.reward_dict[i]['position']
                if is_in_FoV(self.agent.get_state()[:3], self.FoV, self.range, reward_pos):
                    self.reward_dict[i]['collected'] = True
                    reward_collected += 1
        
        return reward_collected
          
    def get_reward(self) -> Tuple[float, bool]:
        reward = - 0.1
        terminated = False

        reward_collected = self.reward_check()
        if reward_collected > 0:
            self.status['reward_collected'] += reward_collected
            reward += reward_collected * 10
        

        if self.agent.boundary_check(self.target_area):
            terminated = True
            self.status['goal'] = True
            reward += 100 
            return reward, terminated   
            
        if not self.agent.boundary_check(self.workspace):
            terminated = True
            self.status['collision'] = True
            reward -= 100
            return reward, terminated
        
        return reward, terminated

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Esegue un passo nell'ambiente in base all'azione fornita.

        Questo metodo aggiorna lo stato del robot in base all'azione scelta, 
        calcola la nuova osservazione, la ricompensa, e verifica se l'episodio è terminato.
        Se la modalità di rendering è impostata su 'human', viene renderizzato un frame.

        Args:
            action (int): L'azione scelta dall'agente.

        Returns:
            observation (np.array): La nuova osservazione dell'ambiente.
            reward (float): La ricompensa ottenuta dopo aver eseguito l'azione.
            terminated (bool): Indica se l'episodio è terminato.
            truncated (bool): Indica se l'episodio è stato troncato.
            info (dict): Informazioni aggiuntive sull'ambiente.
        """
        if not isinstance(action, int):
            action = int(action)

        omega = self.action_to_controls[action]
        reward = 0.0
        # Eseguo l'azione per 4 frame
        for i in range(4):
            self.agent.simple_kinematics(omega=omega, dt=self.dt)
            frame_reward, terminated = self.get_reward()
            reward += frame_reward
            if self.render_mode == 'human':
                self._render_frame()
            if terminated:
                break

        observation = self.get_obs()
        # reward, terminated = self.get_reward()
        truncated = False
        info = self.get_info()

        # if self.render_mode == 'human':
        #     self._render_frame()

        return observation, reward, terminated, truncated, info

    def pygame_init(self) -> None:
        """
        Inizializza pygame per il rendering.

        Questo metodo inizializza la finestra di pygame e il clock se non sono già stati inizializzati.
        Viene chiamato all'inizio del rendering per assicurarsi che pygame sia pronto.
        """
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(self.window_size)
            else: # rgb_array
                self.window = pygame.Surface(self.window_size)

            self.surface_width = self.window_size[0]
            self.surface_height = self.window_size[1]
            self.real_world_canvas = pygame.Surface((self.surface_width, self.surface_height))
            # self.gridmap_canvas = pygame.Surface((self.surface_width, self.surface_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render_real_world(self) -> None:
        """
        Renderizza il mondo reale.

        Questo metodo disegna il robot e la sua direzione sulla canvas del mondo reale.
        Le coordinate del robot vengono convertite in pixel in base alla scala della finestra.
        """
        self.real_world_canvas.fill((255, 255, 255))  # Sfondo bianco

        # Disegno target area
        lenght_in_pixels = round((self.target_area[2] - self.target_area[0]) * self.scale)
        target_area_rect = (
            round((self.target_area[0] - self.min_x) * self.scale),
            round(self.surface_height - (self.target_area[3] - self.min_y) * self.scale),
            lenght_in_pixels,
            lenght_in_pixels
        )

        pygame.draw.rect(
            self.real_world_canvas,
            (255, 0, 0),
            target_area_rect,
            2)
                    
        # Posizione del robot in pixel
        robot_pixel_pos = (
            round((self.agent.x - self.min_x) * self.scale),
            round(self.surface_height - (self.agent.y - self.min_y) * self.scale)
        )

        # Raggio del robot in pixel
        robot_radius_pixels = round(self.robot_radius * self.scale)

        # Disegna il robot
        pygame.draw.circle(
            self.real_world_canvas,
            (0, 0, 255),
            robot_pixel_pos,
            robot_radius_pixels
        )

        # Disegna le ricompense ancora non raccolte
        reward_radius_pixels = round(self.reward_radius * self.scale)

        for i in range(self.n_rewards):
            if not self.reward_dict[i]['collected']:
                reward_pos = (
                    round((self.reward_dict[i]['position'][0] - self.min_x) * self.scale),
                    round(self.surface_height - (self.reward_dict[i]['position'][1] - self.min_y) * self.scale)
                )
                pygame.draw.circle(
                    self.real_world_canvas,
                    (0, 255, 0),
                    reward_pos,
                    reward_radius_pixels
                )
                
        # Disegna il sensore del robot
        sensor_range_pixels = round(self.range * self.scale)
        
        x1 = robot_pixel_pos[0] + round(sensor_range_pixels * np.cos(self.agent.theta - self.FoV/2))
        y1 = robot_pixel_pos[1] - round(sensor_range_pixels * np.sin(self.agent.theta - self.FoV/2))
        x2 = robot_pixel_pos[0] + round(sensor_range_pixels * np.cos(self.agent.theta + self.FoV/2))
        y2 = robot_pixel_pos[1] - round(sensor_range_pixels * np.sin(self.agent.theta + self.FoV/2))

        pygame.draw.line(
            self.real_world_canvas,
            (255, 0, 0),
            robot_pixel_pos,
            (x1, y1),
            1
        )
        pygame.draw.line(
            self.real_world_canvas,
            (255, 0, 0),
            robot_pixel_pos,
            (x2, y2),
            1
        )
        pygame.draw.arc(
            self.real_world_canvas,
            (255, 0, 0),
            (robot_pixel_pos[0] - sensor_range_pixels, robot_pixel_pos[1] - sensor_range_pixels, sensor_range_pixels * 2, sensor_range_pixels * 2),
            self.agent.theta - self.FoV/2,
            self.agent.theta + self.FoV/2,
            1
        )

    def _render_frame(self):
            """
            Renderizza un frame dell'ambiente.

            Questo metodo chiama pygame_init per assicurarsi che pygame sia inizializzato,
            quindi renderizza il mondo reale e la gridmap (se implementata).
            Infine, aggiorna la finestra di pygame con i nuovi disegni.
            """
            self.pygame_init()
            self.render_real_world()

            assert self.window is not None
            assert self.clock is not None

            self.window.blit(self.real_world_canvas, (0, 0))

            if self.render_mode == "human":
                pygame.event.pump()
                pygame.display.flip()
                self.clock.tick(self.metadata['render_fps'])

            elif self.render_mode == "rgb_array":
                return np.transpose(np.array(pygame.surfarray.pixels3d(self.real_world_canvas)), axes=(1, 0, 2))
    
            
    def render(self, mode: str = "human"):
        """
        Renderizza l'ambiente in base alla modalità scelta.

        Args:
            mode (str): La modalità di rendering, può essere 'human' o 'rgb_array'.

        Returns:
            rendered_frame (np.array, opzionale): Un array numpy che rappresenta il frame renderizzato.
        """
        assert mode in self.metadata["render_modes"]
        self.render_mode = mode

        return self._render_frame()
    
    def close(self) -> None:
        """
        Chiude l'ambiente e pygame.

        Questo metodo chiude la finestra di pygame e termina il modulo pygame.
        Viene chiamato quando l'ambiente non è più necessario.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    



if __name__ == "__main__":
    """" DEBUG """
    env = RewardExplorerEnv(render_mode='human')
    observation, info = env.reset()
    done = False
    tot_reward = 0.0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        tot_reward += reward

        print("Observation:", observation)
        print("Reward:", tot_reward)
        print("Done:", done)
        print("Info:", info)