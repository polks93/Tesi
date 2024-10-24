import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
import  numpy as np
from typing import Tuple, Deque, Optional
from collections import deque
import random

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
 
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.init_weights()

    def init_weights(self):

        # Init di kaiming per strati con ReLU
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

        # Init di Xavier per l'ultimo strato con tanh
        nn.init.xavier_uniform_(self.fc3.weight)

            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcola l'azione attraverso il modello. Questa funazione viene eseguita automaticamente quando si chiama il modello.
        Args:
            x (torch.Tensor): Il tensore di input rappresentante lo stato.
        Returns:
            torch.Tensor: L'azione dettata dalla policy.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action = torch.tanh(self.fc3(x))
        
        return action

class Critic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.init_weights()

    def init_weights(self):
        # Init di kaiming per strati con ReLU
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight)
    

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calcola il valore Q attraverso il modello. Questa funzione viene eseguita automaticamente quando si chiama il modello.
        Args:
            state (torch.Tensor): Il tensore di input rappresentante lo stato.
            action (torch.Tensor): Il tensore di input rappresentante l'azione.
        Returns:
            torch.Tensor: Il valore Q.
        """

        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(torch.cat((x, action), dim=1)))
        q_value = self.fc3(x)

        return q_value

class AC_agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        """
        Inizializza l'agente con le reti Actor e Critic, gli ottimizzatori e i parametri necessari.

        Args:
            state_dim (int): Dimensione dello stato.
            action_dim (int): Dimensione dell'azione.
            device (torch.device): Dispositivo su cui eseguire le operazioni (CPU o GPU).
            actor_lr (float): Tasso di apprendimento per l'Actor.
            critic_lr (float): Tasso di apprendimento per il Critic.
            gamma (float): Fattore di sconto per le ricompense future.
            tau (float): Tasso per l'aggiornamento soft delle reti target.
        """      
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Salvo i limiti delle azioni come tensori
        if action_low is None:
            action_low = - np.ones(action_dim)
        if action_high is None:
            action_high = np.ones(action_dim)

        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

        # Inizializza le reti Actor e Critic
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)

        # Inizializza le target networks
        self.target_actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        # Copia i pesi delle reti iniziali nelle target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Inizializza gli ottimizzatori
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.criterion = nn.MSELoss()

    def select_action(self, state: np.ndarray, noise_std: float) -> np.ndarray:
        """
        Seleziona un'azione data dallo stato corrente, aggiungendo rumore per l'esplorazione.

        Args:
            state (np.ndarray): Stato corrente dell'ambiente.
            noise_std (float): Deviazione standard del rumore gaussiano aggiunto all'azione.

        Returns:
            np.ndarray: Azione selezionata, con rumore aggiunto per l'esplorazione.
        """
        
        # Imposta la rete in modalità valutazione
        self.actor.eval()

        # Converte lo stato in un tensore e aggiunge una dimensione
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Calcola l'azione attraverso la rete Actor (tensore)
        with torch.no_grad():
            action = self.actor(state_tensor)

        # Reimposta la rete in modalità addestramento
        self.actor.train()

        # Aggiungi rumore all'azione
        noise = torch.normal(0, noise_std, size=action.size()).to(self.device)
        action += noise

        # Clip dell'azione per rientrare nei limiti definiti
        action = torch.clamp(action, self.action_low, self.action_high)

        return action.cpu().numpy().flatten()
    

    def update(
            self,
            memory: ReplayBuffer,
            batch_size: int,
            debug: bool = False
    ) -> Optional[Tuple[float, float]]:
        """
        Aggiorna le reti Actor e Critic utilizzando un batch di esperienze dal replay buffer.

        Args:
            memory (ReplayBuffer): Il buffer da cui campionare le esperienze.
            batch_size (int): La dimensione del batch da campionare.

        Returns:
            Tuple[float, float]: Le perdite per il Critic e l'Actor.
        """
        # Campiono una serie di batch dal buffer
        states, actions, rewards, next_states, terminateds = memory.sample(batch_size)

        # Converto tutto in tensori e sposta sul device della rete policy_net
        states_tensor       = torch.tensor(states,      dtype=torch.float32).to(self.device)
        actions_tensor      = torch.tensor(actions,     dtype=torch.float32).to(self.device)
        rewards_tensor      = torch.tensor(rewards,     dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_tensor  = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        terminateds_tensor  = torch.tensor(terminateds, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Calcolo le azioni successive e i valori Q target
            next_actions = self.target_actor(next_states_tensor)
            target_q_values = self.target_critic(next_states_tensor, next_actions)
            y = rewards_tensor + self.gamma * target_q_values * (1 - terminateds_tensor)
        
        q_values = self.critic(states_tensor, actions_tensor)
        critic_loss = self.criterion(q_values, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        predicted_actions = self.actor(states_tensor)
        actor_loss = - self.critic(states_tensor, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Aggiornamento delle target networks
        self._update_target_network(self.target_actor, self.actor)
        self._update_target_network(self.target_critic, self.critic)

        # Restituisce le perdite solo se richiesto (questa operazione rallenta l'addestramento)
        if debug:
            return critic_loss.item(), actor_loss.item()
        
        return None
    
    def _update_target_network(
            self,
            target: nn.Module,
            source: nn.Module
    ) -> None:
        """
        Aggiorna i pesi della target network utilizzando l'aggiornamento soft.

        Args:
            target (nn.Module): La rete target da aggiornare.
            source (nn.Module): La rete sorgente da cui copiare i pesi.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

   
if __name__ == '__main__':
    """ Test delle classi Actor e Critic """
    import numpy as np
    state_dim = 2
    action_dim = 1
    hidden_dim = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    states = torch.rand(batch_size, state_dim).to(device)
    actions = torch.rand(batch_size, action_dim).to(device)

    actor = Actor(state_dim, action_dim, hidden_dim).to(device)
    critic = Critic(state_dim, action_dim, hidden_dim).to(device)

    actions_pred = actor(states)
    q_values = critic(states, actions)

    # print(f'Actions: {actions_pred}')
    # print(f'Q values: {q_values}')
    low = np.array([-1.0])
    agent = AC_agent(state_dim, action_dim, hidden_dim, device, action_low=low)
    state = np.random.rand(state_dim)
    action = agent.select_action(state, 0.1)
    # print(f'State: {state}')
    # print(f'Action: {action}')

    memory = ReplayBuffer(1000)
    for i in range(1000):
        state = np.random.rand(state_dim)
        action = np.random.rand(action_dim)
        reward = random.random()
        next_state = np.random.rand(state_dim)
        terminated = random.choice([True, False])
        memory.push(state, action, reward, next_state, terminated)

    agent.update(memory, batch_size)
    selected_actions = []
    for i in range(10000):
        state = np.random.rand(state_dim)
        action = agent.select_action(state, 0.0)
        # print(action)
        selected_actions.append(action)
    print('Update completed')

    import matplotlib.pyplot as plt

    # Crea il plot
    plt.plot(selected_actions)
    plt.xlabel('Step')
    plt.ylabel('First Action Value')
    plt.title('First Action Value over Time')
    plt.show()

    """Test action clamp """
    # import torch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # action_dim = 2
    # action_low = np.array([-2.0, -3.0])
    # action_high = np.array([2.0, 3.0])

    # low_tensor = torch.tensor(action_low, dtype=torch.float32).to(device)
    # high_tensor = torch.tensor(action_high, dtype=torch.float32).to(device)

    # print(low_tensor)
    # print(high_tensor)

    # action = - np.random.rand(action_dim) * 10
    # action_tensor = torch.tensor(action, dtype=torch.float32).to(device)
    # print(action_tensor)
    # action_tensor = torch.clamp(action_tensor, low_tensor, high_tensor)
    # print(action_tensor)
