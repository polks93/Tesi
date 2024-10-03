import numpy as np
import pygame

import  gymnasium as        gym
from    gymnasium import    spaces

# Definizione classe env
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=6, random_goal=True, random_start=True):
        self.size = size
        self.window_size = 512
        self.random_goal = random_goal
        self.random_start = random_start
        self.fixed_start = np.array([0,0])
        self.fixed_goal = np.array([size-1, size-1])

        # Definisco lo spazio delle osservazioni un dizionario contente la posizione dell'agente e quella del target
        self.observation_space = spaces.Dict(
            { 
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # Definisco lo spazio delle azioni, saranno 4 possibili azioni NSWE 
        self.action_space = spaces.Discrete(4)

        # Conversione da azione discreta a movimento
        self._action_to_direction = {
            0: np.array([1,0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }       

        # Imposto la render mode, se passata correttamente altrimenti passo un errore
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    # Funzione per ottenere le osservazioni 
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    # Funzione per ottenere le info: distanza tra agente e target
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    # Funzione reset. Serve per inizializzare l'ambiente all'inizio di ogni nuovo episodio
    def reset(self, seed=None, options=None):
        # super() va a richiamare un attributo della classe genitore gym.Env
        # In questo caso serve ad assegnare un eventuale seed casuale
        super().reset(seed=seed)

        if self.random_start is True:
            if self.random_goal is True:
                # Posizione iniziale agente random
                self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            else:
                # Mi assicuro di non partire mai dal GOAL
                self._agent_location = self.fixed_goal
                while np.array_equal(self.fixed_goal, self._agent_location):
                    self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            self._agent_location = self.fixed_start

        if self.random_goal is True:
            # Posizione del target. Viene campionata in modo da non coincidere con la posizione
            #  dell'agente
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:   
            self._target_location = self.fixed_goal

        # Uso le funzioni dichiarate sopra per ottenre le osservazioni e le info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # Funzione step. Serve a eseguire un ciclo dopo aver deciso un azione, si puo chiamare solo dopo reset
    def step(self, action):
        
        # Ottiene la direzione di movimento dall'azione binaria
        direction = self._action_to_direction[action]
        # Update posizione agente. np.clip serve a restare nel box [0 1 2 3 4]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Ad ogni step ho una reward negativa
        reward = -1

        # Ottengo osservazioni e info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        terminated = np.array_equal(self._agent_location, self._target_location)

        return observation, reward, terminated, False, info


    # Funzione di rendering basata su Pygame
    def render(self):
        if self.render_mode== "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Genero un buffer su cui disegnare l'ambiente prima di disengarlo sullo schermo
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size/self.size)

        # Inizio a disegnare il target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size*self._target_location, 
                (pix_square_size, pix_square_size),
            ),
        )

        # Disegno agente
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Disegno la griglia
        for x in range(self.size + 1):
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
            # Copio il buffer appena creato su una finestra visibile
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            #  Mi assicuro di mantenere il framerate
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))

    # Funzione per chiudere la finestra di rendering
    def close(self):
        if self.window is not None:
            pygame.display.quit()
