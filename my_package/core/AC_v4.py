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
    """
    Inizializza i pesi di un layer con valori casuali all'interno di un intervallo calcolato.

    Args:
        layer (torch.nn.Module): Il layer di cui inizializzare i pesi.
    Returns:
        tuple: Una tupla contenente i limiti inferiore e superiore dell'intervallo di inizializzazione.
    Notes:
        L'intervallo è calcolato come (-1/sqrt(fan_in), 1/sqrt(fan_in)), dove fan_in è il numero di unità di input nel layer.
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """
    Classe Actor che rappresenta una rete neurale per apprendimento per rinforzo.

    Args:
        state_size (int): Dimensione dello spazio degli stati.
        action_size (int): Dimensione dello spazio delle azioni.
        network_params (dict): Dizionario dei parametri della rete, deve includere 'fc1' e 'fc2'.

    Methods:
        forward(state: torch.Tensor) -> torch.Tensor:
            Propaga l'input attraverso la rete e restituisce le azioni.
        add_param_noise(scalar: float = 1.0):
            Aggiunge rumore ai pesi della rete per favorire l'esplorazione.
    """

    def __init__(self, state_size: int, action_size: int, network_params: dict) -> None:
        super(Actor, self).__init__()

        if not 'fc1' in network_params or not 'fc2' in network_params:
            raise ValueError("Missing network parameters. Must provide 'fc1' and 'fc2' values.")
        fc1_units: int = network_params['fc1']
        fc2_units: int = network_params['fc2']

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
    """
    Classe Critic che costruisce una rete neurale per stimare il valore Q di una coppia 
    stato-azione in un algoritmo Actor-Critic.

    Args:
        state_size (int): Dimensione dello stato di input.
        action_size (int): Dimensione dell'azione di input.
        network_params (dict): Dizionario contenente i parametri della rete, 
        deve includere le chiavi 'fc1' e 'fc2'.

    Exceptions:
        ValueError: Se 'fc1' o 'fc2' non sono presenti in `network_params`.
        
    Methods:
        forward(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            Esegue il passaggio in avanti della rete, combinando lo stato e l'azione 
            per restituire il valore Q corrispondente.
    """


    def __init__(self, state_size: int, action_size: int, network_params: dict) -> None:
        super(Critic, self).__init__()
        
        if not 'fc1' in network_params or not 'fc2' in network_params:
            raise ValueError("Missing network parameters. Must provide 'fc1' and 'fc2' values.")
        fc1_units: int = network_params['fc1']
        fc2_units: int = network_params['fc2']

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
   
    def __init__(self, size: int, noise_params: dict) -> None:
        """
        Inizializza il processo di rumore Ornstein-Uhlenbeck.
        Parametri:
        - size: int = La dimensione del vettore di rumore.
        - noise_params: dict = Un dizionario contenente i parametri del rumore OU, ovvero 'mu', 'theta' e 'sigma'.
        Ritorno:
        - None
        """
        if not 'mu' in noise_params or not 'theta' in noise_params or not 'sigma' in noise_params:
            raise ValueError("Missing noise parameters. Must provide 'mu', 'theta' and 'sigma' values.")
        
        mu: float       = noise_params['mu']
        theta: float    = noise_params['theta']
        sigma: float    = noise_params['sigma']

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma

        self.reset()
        
    def reset(self) -> None:
        """
        Reimposta il rumore al valore iniziale.
        Ritorno:
        - None
        """
        self.state = copy.copy(self.mu)

            
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

class ParamNoise:
    """
    Classe per la gestione del rumore parametrico dei pesi della rete neurale actor.
    Attributes:
        desired_std (float): La deviazione standard desiderata tra l'azione con rumore e l'azione senza rumore.
        scalar (float): Il fattore di perturbazione dei pesi.
        scalar_decay (float): Il fattore di decadimento per il fattore di perturbazione dei pesi.
        actor_noised (Actor): La rete actor a cui applicare il rumore parametrico.
    Methods:
        __init__(state_size: int, action_size: int, actor_params: dict, params: dict) -> None:
            Inizializza la classe ParamNoise con i parametri forniti.
        sample(state_tensor: torch.Tensor, action: np.ndarray, actor_network: Actor) -> np.ndarray:
            Applica il rumore parametrico ai pesi dell'actor e restituisce l'azione con rumore.
    """

    def __init__(self, state_size: int, action_size: int, actor_params: dict, params: dict, device: torch.device) -> None:

        if not 'desired_std' in params or not 'scalar' in params or not 'scalar_decay' in params:
            raise ValueError("Missing noise parameters. Must provide 'desired_std', 'scalar' and 'scalar_decay' values.")
     
        self.desired_std: float         = params['desired_std']
        self.scalar: float              = params['scalar']
        self.scalar_decay: float        = params['scalar_decay'] 

        if 'decay_with_ep' in params and params['decay_with_ep']:
            self.decay_with_ep = True
            self.max_std: float = params['desired_std']
            self.min_std: float = params['min_std']
        else:
            self.decay_with_ep = False

        self.actor_noised               = Actor(state_size, action_size, actor_params).to(device)
        self.actor_noised.eval()

    def sample(self, state_tensor: torch.Tensor, action: np.ndarray, actor_network: Actor):
        self.actor_noised.load_state_dict(actor_network.state_dict().copy())
        self.actor_noised.add_param_noise(self.scalar)
        action_noised = self.actor_noised(state_tensor).cpu().numpy().flatten()
        distance = np.sqrt(np.mean(np.square(action - action_noised)))

        if distance > self.desired_std:
            self.scalar *= self.scalar_decay
        if distance < self.desired_std:
            self.scalar *= 1/self.scalar_decay

        return action_noised

class AC_agent():
    """
    Classe agente Actor-Critic-v4.
    Args:
        state_size (int): Dimensione dello stato.
        action_size (int): Dimensione dell'azione.
        device (torch.device): Dispositivo su cui eseguire i calcoli (CPU o GPU).
        noise_params (dict): Parametri per il rumore.
            ou_noise_params = {'mu': 0.0, 'theta': 0.15, 'sigma': 0.2},
            normal_noise_params = {'scalar': 0.25},
            param_noise_params = {'desired_std': 0.7, 'scalar': 0.05, 'scalar_decay': 0.99},
        noise_type (str): Tipo di rumore ('ou', 'param', 'normal').
        actor_params (dict, opzionale): Parametri per la rete dell'attore. Default è {'fc1': 256, 'fc2': 128}.
        critic_params (dict, opzionale): Parametri per la rete del critico. Default è {'fc1': 256, 'fc2': 128}.
        AC_params (dict, opzionale): Parametri per l'algoritmo Actor-Critic. Default è {'learn_every': 16, 'n_learn': 16, 'gamma': 0.99, 'tau': 1e-3, 'lr_actor': 1e-3, 'lr_critic': 1e-3}.
        buffer_params (dict, opzionale): Parametri per il buffer di replay. Default è {'buffer_size': int(1e6), 'batch_size': 128}.
    Raises:
        ValueError: Se mancano parametri necessari.
    """
    metadata = {'noise_types': ['param', 'ou', 'normal']}

    def __init__(

            self, 
            state_size: int,
            action_size: int,
            max_episodes: int,
            device: torch.device,
            noise_params: dict, 
            noise_type: str,
            actor_params: dict = {'fc1': 256, 'fc2': 128},
            critic_params: dict = {'fc1': 256, 'fc2': 128},
            AC_params: dict = {'learn_every': 16, 'n_learn': 16, 'gamma': 0.99, 'tau': 1e-3, 'lr_actor': 1e-3, 'lr_critic': 1e-3},
            buffer_params: dict = {'buffer_size': 1000000, 'batch_size': 128}
    ) -> None:
        
        self.device         = device
        self.state_size     = state_size
        self.action_size    = action_size
        self.max_episodes   = max_episodes

        if 'learn_every' not in AC_params or 'n_learn' not in AC_params or 'gamma' not in AC_params or 'tau' not in AC_params or 'lr_actor' not in AC_params or 'lr_critic' not in AC_params:
            raise ValueError("Missing AC parameters. Must provide 'learn_every', 'n_learn', 'gamma', 'tau', 'lr_actor' and 'lr_critic' values.")
        self.learn_every: int   = AC_params['learn_every']
        self.n_learn: int       = AC_params['n_learn']
        self.gamma: float       = AC_params['gamma']
        self.tau: float         = AC_params['tau']

        if 'batch_size' not in buffer_params or 'buffer_size' not in buffer_params:
            raise ValueError("Missing buffer parameters. Must provide 'batch_size' and 'buffer_size' values.")
        self.batch_size: int    = buffer_params['batch_size']
        self.buffer_size: int   = buffer_params['buffer_size']
        self.memory             = ReplayBuffer(self.buffer_size)


        # Actor Network 
        self.actor              = Actor(state_size, action_size, actor_params).to(device)
        self.actor_target       = Actor(state_size, action_size, actor_params).to(device)
        self.actor_optimizer    = optim.Adam(self.actor.parameters(), lr=AC_params['lr_actor'])
        
        # Critic Network
        self.critic             = Critic(state_size, action_size, critic_params).to(device)
        self.critic_target      = Critic(state_size, action_size, critic_params).to(device)
        self.critic_optimizer   = optim.Adam(self.critic.parameters(), lr=AC_params['lr_critic'])

        # Copio i pesi del modello iniziale nei target
        self.soft_update(self.actor, self.actor_target, 1.0)
        self.soft_update(self.critic, self.critic_target, 1.0)


        if noise_type not in self.metadata['noise_types']:
            raise ValueError(f"Invalid noise type. Must be one of {self.metadata['noise_types']}")
        self.noise_type = noise_type

        if noise_type == 'ou':    
            self.ou_noise = OUNoise(size=action_size, noise_params=noise_params)

        elif noise_type == 'param':
            self.param_noise = ParamNoise(state_size, action_size, actor_params, noise_params, device)

        elif noise_type == 'normal':
            if 'scalar' not in noise_params:
                raise ValueError("Missing noise parameters. Must provide 'scalar' value.")
            self.normal_scalar = noise_params['scalar']

        else:
            raise ValueError(f"Invalid noise type. Must be one of {self.metadata['noise_types']}")
        
        self.counter = 0 

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
        next_actions    = self.actor_target(next_states_tensor)
        Q_targets_next  = self.critic_target(next_states_tensor, next_actions)
        # Calcolo Q_targets per lo stato corrente
        Q_targets = rewards_tensor + self.gamma * Q_targets_next * (1 - terminateds_tensor)
        # Calcolo la loss del critic
        Q_expected  = self.critic(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimizzo la loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- Aggiorno l'attore ---------------------------- #
        actions_pred    = self.actor(states_tensor)
        actor_loss      = - self.critic(states_tensor, actions_pred).mean()
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

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
            if add_noise:
                if self.noise_type == 'param':
                    action_noised = self.param_noise.sample(state_tensor, action, self.actor)

                elif self.noise_type == 'ou':
                    noise   = self.ou_noise.sample()
                    action_noised = action + noise
                    
                elif self.noise_type == 'normal':
                    action_noised = action + self.normal_scalar * np.random.randn(self.action_size)

                action = action_noised
        self.actor.train()
        return np.clip(action, -1.0, 1.0)

    def reset(self, episode: int) -> None:
        if self.noise_type == 'ou':
            self.ou_noise.reset()

        elif self.noise_type == 'param' and self.param_noise.decay_with_ep:
            max_std         = self.param_noise.max_std
            min_std         = self.param_noise.min_std
            max_episodes    = self.max_episodes
            new_std         = np.max([min_std, max_std *(1 - episode / max_episodes)])
            self.param_noise.desired_std = new_std


if __name__ == '__main__':
    import numpy as np
    state_dim = 10
    action_dim = 1
    batch_size = 128
    buffer_size = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AC_params           = {'learn_every': 16, 'n_learn': 16, 'gamma': 0.99, 'tau': 1e-3, 'lr_actor': 1e-3, 'lr_critic': 1e-3}
    buffer_params       = {'buffer_size': int(1e6), 'batch_size': 128}
    param_noise_params        = {'desired_std': 0.5, 'scalar': 0.05, 'scalar_decay': 0.99, 'min_std': 0.01, 'decay_with_ep': True}
    ou_noise_params           = {'mu': 0.0, 'theta': 0.15, 'sigma': 0.2}
    normal_noise_params       = {'scalar': 0.25}

    agent = AC_agent(
        state_size=state_dim, 
        action_size=action_dim, 
        max_episodes=300,
        device=device, 
        noise_params=param_noise_params,
        noise_type='param',
        AC_params=AC_params,
        buffer_params=buffer_params)
    
    selected_actions = []
    # std = []
    # noise = []
    for ep in range(300):
        agent.reset(ep)
        # agent.param_noise.desired_std *= 0.99
        # std.append(agent.ou_noise.sigma)
        # noise.append(agent.ou_noise.sample())
        for step in range(10):
            state = np.random.rand(state_dim)
            action = agent.act(state, add_noise=False)
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
