import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
import  numpy as np
from typing import Tuple, Deque, Optional
from collections import deque
import random
import copy

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """
        Init del buffer per memorizzare le esperienze deque generare una coda a dimensione fissa.
        Quando la dimensione supera la capcità si comporta in modo FIFO e scarto gli elementi
        più vecchi dal buffer
        Parametri:
        - capacity: int = La dimensione massima del buffer.
        Ritorno:
        - None
        """

        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        """
        Aggiunge una nuova esperienza < S, A, R, S', terminated > nel buffer.
        Parametri:
        - state: np.ndarray = Lo stato corrente dell'ambiente.
        - action: np.ndarray = Il vettore delle azioni eseguite nello stato corrente.
        - reward: float = La ricompensa ottenuta eseguendo l'azione nello stato corrente.
        - next_state: np.ndarray = Lo stato successivo dell'ambiente dopo aver eseguito l'azione.
        - terminated: bool = Indica se l'episodio è terminato dopo aver eseguito l'azione.
        Ritorno:
        - None
        """

        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Campiona un batch di esperienze dal buffer. Il batch contiene tanta esperienze quante definite in batch_size.
        Args:
            batch_size (int): La dimensione del batch da campionare.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Una tupla contenente gli array NumPy dei
            batch di stati, azioni, ricompense, prossimi stati e flag di terminazione.
        """

        states, actions, rewards, next_states, terminateds = zip(*random.sample(self.buffer, batch_size))
        
        # Converto ogni elemento in array NumPy per facilitarne l'uso in PyTorch
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminateds)

    def __len__(self) -> int:
        """
        Restituisce il numero di esperienze attualmente memorizzate nel buffer.
        """
        return len(self.buffer)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size: int, action_size: int, fc1_units: int = 256, fc2_units: int = 128) -> None:
        super(Actor, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.BatchNorm1d(fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size),
            nn.Tanh()
        )

        # Inizializzazione pesi della rete
        self.seq[0].weight.data.uniform_(*hidden_init(self.seq[0]))
        self.seq[3].weight.data.uniform_(*hidden_init(self.seq[3]))
        self.seq[5].weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.seq(state)
    
    def add_param_noise(self, scalar: float = 1.0):
        for layer in [0, 3, 5]:
            self.seq[layer].weight.data += scalar * torch.randn_like(self.seq[layer].weight.data)

class Critic(nn.Module):
    def __init__(self, state_size: int, action_size: int, fc1_units: int = 256, fc2_units: int = 128) -> None:
        super(Critic, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.BatchNorm1d(fc1_units),
            nn.ReLU()
            )
        
        self.seq2 = nn.Sequential(
            nn.Linear(fc1_units + action_size, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )

        # Inizializzazione pesi della rete
        self.seq1[0].weight.data.uniform_(*hidden_init(self.seq1[0]))
        self.seq2[0].weight.data.uniform_(*hidden_init(self.seq2[0]))
        self.seq2[2].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.seq1(state)
        x = torch.cat((x, action), dim=1)
        return self.seq2(x)

class OUNoise:

    def __init__(
            self, 
            size: int,
            noise_decay: bool = False,
            max_episodes: Optional[int] = None,
            decay_starts: int = 0,
            sigma_max: float = 0.2,
            sigma_min: float = 0.05, 
            mu: float = 0.0, 
            theta: float = 0.15 
            ) -> None:
        """
        Inizializza il processo di rumore Ornstein-Uhlenbeck.
        Parametri:
        - size: int = La dimensione del vettore di rumore.
        - mu: float = Il parametro di media del rumore.
        - theta: float = Il parametro theta del rumore.
        - sigma: float = Il parametro sigma del rumore.
        Ritorno:
        - None
        """

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma_max
        self.noise_decay = noise_decay

        if self.noise_decay:
            self.sigma_max = sigma_max
            self.sigma_min = sigma_min
            self.noise_decay = noise_decay
            self.max_episodes = max_episodes
            self.decay_starts = decay_starts

        self.state = copy.copy(self.mu)
        
    def reset(self, episode: Optional[int] = None) -> None:
        """
        Reimposta il rumore al valore iniziale.
        Ritorno:
        - None
        """
        self.state = copy.copy(self.mu)
        if self.noise_decay and episode is not None:
            assert self.max_episodes is not None, "Max episodes must be provided for decayed noise"
            if episode > self.decay_starts:
                # Decadimento lineare del valore di sigma
                self.sigma = self.sigma_max - (self.sigma_max - self.sigma_min) * (episode - self.decay_starts) / (self.max_episodes - self.decay_starts)

            
    def sample(self) -> np.ndarray:
        """
        Campiona un rumore dal processo di Ornstein-Uhlenbeck.
        Ritorno:
        - np.ndarray: Il rumore campionato dal processo.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state
        

class AC_agent():
    metadata = {'noise_types': ['param', 'ou']}
    def __init__(
            self, 
            state_size: int,
            action_size: int,
            device: torch.device,
            noise_type: str = 'ou',
            noise_decay: bool = False,
            max_episodes: Optional[int] = None,
            decay_starts: int = 0,
            sigma_max: float = 0.2,
            sigma_min: float = 0.01,
            learn_every: int = 16,
            n_learn: int = 16,
            gamma: float = 0.99,
            tau: float = 1e-3,
            lr_actor: float = 1e-3,
            lr_critic: float = 1e-3,
            batch_size: int = 128,
            buffer_size: int = int(1e6),
            scalar: float = .05,
            scalar_decay: float = .99,
            normal_scalar: float = .25,
            desired_distance: float = .7
    ) -> None:
        
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

        self.learn_every = learn_every
        self.n_learn = n_learn
        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size

        if noise_type not in self.metadata['noise_types']:
            raise ValueError(f"Invalid noise type. Must be one of {self.metadata['noise_types']}")
        self.noise_type = noise_type
        self.desired_distance = desired_distance
        self.scalar = scalar
        self.scalar_decay = scalar_decay
        self.normal_scalar = normal_scalar

        self.counter = 0

        # Actor Network 
        self.actor              = Actor(state_size, action_size).to(device)
        self.actor_target       = Actor(state_size, action_size).to(device)
        self.actor_optimizer    = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.actor_noised       = Actor(state_size, action_size).to(device)

        # Critic Network
        self.critic             = Critic(state_size, action_size).to(device)
        self.critic_target      = Critic(state_size, action_size).to(device)
        self.critic_optimizer   = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Copio i pesi del modello iniziale nei target
        self.soft_update(self.actor, self.actor_target, 1.0)
        self.soft_update(self.critic, self.critic_target, 1.0)

        if noise_type == 'ou':
            self.ou_noise = OUNoise(
                size=action_size, 
                noise_decay=noise_decay,
                max_episodes=max_episodes,
                decay_starts=decay_starts, 
                sigma_max=sigma_max, 
                sigma_min=sigma_min
                )

        self.memory = ReplayBuffer(buffer_size)


    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float) -> None:
        """
        Aggiorna i pesi del target network con un'aggiornamento soft.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Parametri:
        - local_model: nn.Module = Il modello locale da cui copiare i pesi.
        - target_model: nn.Module = Il modello target in cui copiare i pesi.
        - tau: float = Il fattore di aggiornamento soft.
        Ritorno:
        - None
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        """
        Aggiunge una nuova esperienza al buffer e aggiorna i pesi della rete.
        Parametri:
        - state: np.ndarray = Lo stato corrente dell'ambiente.
        - action: np.ndarray = Il vettore delle azioni eseguite nello stato corrente.
        - reward: float = La ricompensa ottenuta eseguendo l'azione nello stato corrente.
        - next_state: np.ndarray = Lo stato successivo dell'ambiente dopo aver eseguito l'azione.
        - terminated: bool = Indica se l'episodio è terminato dopo aver eseguito l'azione.
        Ritorno:
        - None
        """

        self.counter += 1
        self.memory.push(state, action, reward, next_state, terminated)

        if not (self.counter % self.learn_every == 0 and len(self.memory) > self.batch_size):
            return
    
        for _ in range(self.n_learn):
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)


    def learn(
            self, 
            experiences: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ) -> None:
        """
        Aggiorna i pesi della rete utilizzando un batch di esperienze campionate dal buffer.
        Parametri:
        - experiences: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] = Una tupla contenente gli array NumPy dei
            batch di stati, azioni, ricompense, prossimi stati e flag di terminazione.
        Ritorno:
        - None
        """

        states, actions, rewards, next_states, terminateds = experiences

        # Converto tutto in tensori e sposta sul device della rete policy_net
        states_tensor       = torch.tensor(states,      dtype=torch.float32).to(self.device)
        actions_tensor      = torch.tensor(actions,     dtype=torch.float32).to(self.device)
        rewards_tensor      = torch.tensor(rewards,     dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_tensor  = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        terminateds_tensor  = torch.tensor(terminateds, dtype=torch.float32).unsqueeze(1).to(self.device)

        # ---------------------------- Aggiorno il critic ---------------------------- #
        next_actions = self.actor_target(next_states_tensor)
        Q_targets_next = self.critic_target(next_states_tensor, next_actions)
        # Calcolo Q_targets per lo stato corrente
        Q_targets = rewards_tensor + self.gamma * Q_targets_next * (1 - terminateds_tensor)
        # Calcolo la loss del critic
        Q_expected = self.critic(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimizzo la loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- Aggiorno l'attore ---------------------------- #
        actions_pred = self.actor(states_tensor)
        actor_loss = - self.critic(states_tensor, actions_pred).mean()
        # Minimizzo la loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------------- Aggiorno i target networks ---------------------------- #
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)
    
    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        self.actor_noised.eval()

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
            if add_noise:
                if self.noise_type == 'param':
                    self.actor_noised.load_state_dict(self.actor.state_dict().copy())
                    self.actor_noised.add_param_noise(self.scalar)
                    action_noised = self.actor_noised(state_tensor).cpu().numpy().flatten()

                    distance = np.sqrt(np.mean(np.square(action-action_noised)))

                    if distance > self.desired_distance:
                        self.scalar *= self.scalar_decay
                    if distance < self.desired_distance:
                        self.scalar *= 1/self.scalar_decay
                    
                    action = action_noised
                
                elif self.noise_type == 'ou':
                    noise = self.ou_noise.sample()
                    action += noise
                    
                else:
                    action += self.normal_scalar * np.random.randn(self.action_size)

        self.actor.train()
        return np.clip(action, -1.0, 1.0)

    def reset(self, episode: Optional[int] = None) -> None:
        if self.noise_type == 'ou':
            self.ou_noise.reset(episode)



if __name__ == '__main__':
    import numpy as np
    state_dim = 10
    action_dim = 1
    batch_size = 128
    buffer_size = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = AC_agent(
        state_dim, 
        action_dim, 
        device, 
        noise_type='param', 
        noise_decay=False,
        max_episodes=300,
        decay_starts=1,
        desired_distance=.01,
        scalar_decay=.999,
        buffer_size=buffer_size)
    
    selected_actions = []
    # std = []
    # noise = []
    for ep in range(300):
        agent.reset(ep)
        # std.append(agent.ou_noise.sigma)
        # noise.append(agent.ou_noise.sample())
        for step in range(2000):
            state = np.random.rand(state_dim)
            action = agent.act(state)
            selected_actions.append(action)
            # reward = random.random()
            # next_state = np.random.rand(state_dim)
            # terminated = random.choice([True, False])
            # agent.step(state, action, reward, next_state, terminated)

    import matplotlib.pyplot as plt

    # Crea il plot
    plt.plot(selected_actions)
    plt.xlabel('Step')
    plt.ylabel('First Action Value')
    plt.title('First Action Value over Time')
    plt.show()
