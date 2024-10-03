import numpy as np
import pygame
import  gymnasium as        gym
from    gymnasium import    spaces

class CliffEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]} 

    def __init__(self, render_mode=None, fps=10, spawn_reward=False, n_reward=3, reward_seed=None, reward_position=None) -> None:
        # Dati rendering
        self.window_width = 600
        self.window_height = 200
        self.fps = fps
        # Info reward casuali sporadiche
        self.spawn_reward = spawn_reward
        self.reward_dict = {}
        if reward_position is None:
            self.n_reward = n_reward
        else:
            self.n_reward = len(reward_position)
        self.reward_seed = reward_seed
        self.reward_position = reward_position
        # Posizione stato terminale
        self._target_location = np.array([3,11])
       
        # Definizione spazio delle osservazioni e delle azioni
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 11]), dtype=np.int32)
        self.action_space = spaces.Discrete(4)

        # Conversione da azione discreta a movimento
        self._action_to_direction = {
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

    def get_info(self):
        goal_reached = np.array_equal(self._agent_location, self._target_location)
        if self.spawn_reward:
            reward_collected = 0
            for i in range(self.n_reward):
                if self.reward_dict[i]['collected']:
                    reward_collected += 1
            return {"goal": goal_reached, "reward_collected": reward_collected}
        else:
            return {"goal": goal_reached}
    
    # Funzione per generare il dizionario delle reward
    def generate_reward(self):
        np.random.seed(self.reward_seed)
        reward_dict = {} 
        unique_value = set() # Set per evitare duplicati

        # Generazione reward casuali in posizioni uniche
        if self.reward_position is None:
            for i in range(self.n_reward):
                while True:
                    value = (np.random.randint(0, 3), np.random.randint(0, 12))
                    if value not in unique_value:  
                        unique_value.add(value)
                        reward_dict[i] = {'position': np.array([value[0], value[1]]), 'collected': False}
                        break

        # Generazione reward in posizioni specificate dall'utente
        else:
            for idx, reward in enumerate(self.reward_position):
                reward_dict[idx] = {'position': np.array([reward[0], reward[1]]), 'collected': False}  

        return reward_dict

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([3,0])
        observation = self._agent_location

        # Genero il dizionario delle reward
        if self.spawn_reward:
            self.reward_dict = self.generate_reward()

        if self.render_mode == "human":
            self._render_frame()

        info = self.get_info()
        return observation, info
    
    def get_reward(self):
        terminated = False
        reward = -1
        # Falling from the cliff
        if self._agent_location[0] == 3 and self._agent_location[1] > 0 and self._agent_location[1] < 11:
            reward = -100
            terminated = True

        # Goal reached
        elif np.array_equal(self._agent_location, self._target_location):
            if self.spawn_reward:
                for i in range(self.n_reward):
                    if self.reward_dict[i]['collected']:
                        reward += 10
            terminated = True

        # Sporadic Reward
        elif self.spawn_reward:
            for i in range(self.n_reward):
                if np.array_equal(self._agent_location, self.reward_dict[i]['position']) and not self.reward_dict[i]['collected']:
                    self.reward_dict[i]['collected'] = True
                    reward = 10

        return reward, terminated
    
    def step(self, action):
        
        direction = self._action_to_direction[action]
        # Mi assicuro che l'agente resti nel box
        self._agent_location = np.clip(self._agent_location + direction, [0, 0], [3, 11])

        reward, terminated = self.get_reward()
        observation = self._agent_location
        info = self.get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    # Funzione di rendering basata su Pygame
    def render(self):
        if self.render_mode== "rgb_array":
            return self._render_frame()
    

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        target = np.array([self._target_location[1], self._target_location[0]])
        agent = np.array([self._agent_location[1], self._agent_location[0]])   

        # Genero un buffer su cui disegnare l'ambiente prima di disengarlo sullo schermo
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        pix_square_size = round(self.window_width/12)

        # Disegno la cliff
        pygame.draw.line(
            canvas,
            0,
            (pix_square_size, pix_square_size * 7/2), 
            (pix_square_size * 11, pix_square_size * 7/2),
            width=pix_square_size
        )

        # Disegno il target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size*target, 
                (pix_square_size, pix_square_size),
            ),
        )

        # Disegno agente
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (agent + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Disegno sporadic reward
        if self.spawn_reward:
            for i in range(self.n_reward):
                if self.reward_dict[i]['collected']:
                    continue
                reward_position = np.array([self.reward_dict[i]['position'][1], self.reward_dict[i]['position'][0]])
                pygame.draw.circle(
                    canvas,
                    (0, 255, 0),
                    (reward_position + 0.5) * pix_square_size,
                    pix_square_size / 3,
                )

        # Disegno la griglia
        for x in range(4):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_width, pix_square_size * x),
                width=3,
            )
        for y in range(12):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * y, 0),
                (pix_square_size * y, self.window_height),
                width=3,
            )

        if self.render_mode == "human":
            # Copio il buffer appena creato su una finestra visibile
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            #  Mi assicuro di mantenere il framerate
            self.clock.tick(self.fps)
        else: # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))

    # Funzione per chiudere la finestra di rendering
    def close(self):
        if self.window is not None:
            pygame.display.quit()
