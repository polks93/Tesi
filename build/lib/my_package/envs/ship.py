
import pygame
import warnings
import numpy        as      np
import gymnasium    as      gym
from   gymnasium    import  spaces


class ShipEnv2D(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10} 

    def __init__(self, render_mode=None, grid_size=12, ship_size=[3,4]) -> None:
        self.render_mode = render_mode
        self.window_size = 512
        self.grid_size = grid_size

        # Controllo il tipo e la lunghezza di ship_size
        if not isinstance(ship_size, list) or len(ship_size) != 2:
            warnings.warn("Le dimensioni della nave devono essere una lista [i, j]. Imposto le dimensioni predefinite [3, 4].")
            self.ship_size = np.array([3, 4])
        elif ship_size[0] >= grid_size - 2 or ship_size[1] >= grid_size - 2:
            warnings.warn("Le dimensioni della nave sono troppo grandi rispetto alla griglia. Imposto le dimensioni predefinite [3, 4].")
            self.ship_size = np.array([3, 4])
        else:
            self.ship_size = np.array(ship_size)

        # Definisco lo spazio delle osservazioni e delle azioni
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, grid_size - 1, shape=(2,), dtype=np.integer)

        # Conversione da azione discreta a movimento
        self.action_to_direction = {
            0: np.array([1,0]),       # Giu
            1: np.array([0, 1]),      # Destra
            2: np.array([-1, 0]),     # Su
            3: np.array([0, -1]),     # Sinistra
        } 

        # Imposto la render mode, se passata correttamente altrimenti passo un errore
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    # Funzione che genera l'array contenente le celle occupate dalla nave
    def generate_ship(self):

        start_row = (self.grid_size - self.ship_size[0]) // 2
        start_col = (self.grid_size - self.ship_size[1]) // 2
        coordinates = []
        for i in range(self.ship_size[0]):
            for j in range(self.ship_size[1]):
                coordinates.append([start_row + i, start_col + j])

        return np.array(coordinates)
    
    # Funzione che genera il dizionario contenente le reward
    def spawn_reward(self):
        # Salvo i 4 angoli dell'ostacolo
        min_row = np.min(self.ship_location[:,0])
        max_row = np.max(self.ship_location[:,0])
        min_col = np.min(self.ship_location[:,1])
        max_col = np.max(self.ship_location[:,1])
        n = 0
        reward_dict = {}
        # Genero le reward attorno alla nave
        for i in range(min_row -1, max_row + 2):
            for j in ([min_col - 1, max_col + 1]):
                reward_dict[n] = {'position': np.array([i,j]), 'collected': False}
                n += 1
        for j in range(min_col, max_col + 1):
            for i in ([min_row - 1, max_row + 1]):
                reward_dict[n] = {'position': np.array([i,j]), 'collected': False}
                n += 1

        return reward_dict

    # Funzione che restituisce l'osservazione dell'agente
    def get_observation(self):
        return self.agent_location
    
    # Reset dell'ambiente
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Inizializzo lo status agente
        self.status = {'goal': False, 'reward_collected': 0, 'collision': False}
        
        # Inizializzo la posizione dell'agente e dello stato terminale
        self.agent_location = np.array([0,0])
        self.target_location = np.array([0,0])

        # Inizializzo la posizione della nave
        self.ship_location = self.generate_ship()

        # Genero le reward attorno alla nave
        self.reward_dict = self.spawn_reward()
        self.n_reward = len(self.reward_dict)
        
        # Genero le osservazioni e le info
        observation = self.get_observation()
        info = self.status

        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    # Funzione che controlla se la nave collide con l'agente
    def collision_check(self):
        for cell in self.ship_location:
            if np.array_equal(self.agent_location, cell):
                return True
        return False
    
    # Funzione che controlla se l'agente ha raggiunto una reward che non è ancora stata raccolta
    def reward_check(self):
        for i in range(self.n_reward):
            if np.array_equal(self.agent_location, self.reward_dict[i]['position']) and not self.reward_dict[i]['collected']:
                self.reward_dict[i]['collected'] = True
                return True
        return False
    
    def get_reward(self):

        # Collision check
        if self.collision_check():
            reward = -100
            terminated = True
            self.status['collision'] = True

        # Goal reached
        elif np.array_equal(self.agent_location, self.target_location):
            # Se non ho raccolto nessuna reward, non termino l'episodio
            if self.status['reward_collected'] == 0:
                reward = -1
                terminated = False
            # Se ho raccolto tutte le reward, termino l'episodio con reward massima
            elif self.status['reward_collected'] == self.n_reward:
                reward = 100
                terminated = True
                self.status['goal'] = True
            # Se ho raccolto più della metà delle reward, termino l'episodio con reward intermedia
            elif self.status['reward_collected'] > self.n_reward / 2:
                reward = 60
                terminated = True
                self.status['goal'] = True
            # Se ho raccoltola metà o meno reward, termino l'episodio con reward minima
            else:
                reward = 30
                terminated = True
                self.status['goal'] = True

        # Reward collected
        elif self.reward_check():
            reward = 5
            terminated = False
            self.status['reward_collected'] += 1
            
        # Normal step
        else:
            reward = -1
            terminated = False
        return reward, terminated
    
    # Funzione che esegue un azione nell'ambiente
    def step(self, action):
        
        # Eseguo l'azione e mi assicuro di restare nella griglia
        direction = self.action_to_direction[action]
        self.agent_location = np.clip(self.agent_location + direction, 0, self.grid_size - 1)

        # Ottengo la reward e lo stato terminale
        reward, terminated = self.get_reward()

        # Calcolo observation e info 
        observation = self.get_observation()
        info = self.status

        # Rendering frame aggiornato
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    # Funzione per disegnare un quadrato
    def draw_rect(self, canvas, cell, color, scale_factor=1):
        x = cell[1]
        y = cell[0]
        pix_square_size = round(self.window_size / self.grid_size)
        pygame.draw.rect(
            canvas,
            color,
            pygame.Rect(
                (pix_square_size * x, pix_square_size * y),
                scale_factor*(pix_square_size, pix_square_size),
            ),
        )
    
    # Funzione per disegnare un cerchio
    def draw_circle(self, canvas, cell, color, scale_factor: float=1.0):
        x = cell[1]
        y = cell[0]
        pix_square_size = round(self.window_size / self.grid_size)
        pygame.draw.circle(
            canvas,
            color,
            ((x + 0.5) * pix_square_size, (y + 0.5) * pix_square_size),
            scale_factor * pix_square_size / 3,
        )

    def _render_frame(self):
        # Init pygame
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Genero un buffer su cui disegnare l'ambiente prima di disengarlo sullo schermo
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = round(self.window_size / self.grid_size)

        # Disegno la nave
        for cells in self.ship_location:
            self.draw_rect(canvas, cells, (0, 0, 0))

        # Disegno il target
        self.draw_rect(canvas, self.target_location, (255, 0, 0))

        # Disegno agente
        self.draw_circle(canvas, self.agent_location, (0, 0, 255))

        # Disegno sporadic reward
        for i in range(self.n_reward):
            if self.reward_dict[i]['collected']:
                continue
            self.draw_circle(canvas, self.reward_dict[i]['position'], (0, 255, 0), scale_factor=0.8)

        # Disegno la griglia
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":

            assert self.window  is not None
            assert self.clock   is not None

            # Copio il buffer appena creato su una finestra visibile
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            #  Mi assicuro di mantenere il framerate
            self.clock.tick(self.metadata['render_fps'])
        else: # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))
        

    # Funzione per chiudere la finestra di rendering
    def close(self):
        if self.window is not None:
            pygame.display.quit()


if __name__ == "__main__":
    env = ShipEnv2D(render_mode="human")
    observation, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        observation, reward, done, _, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

    env.close()