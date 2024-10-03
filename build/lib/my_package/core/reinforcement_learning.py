import  torch
import  torch.nn as nn
from    torch.optim.optimizer import Optimizer
import  numpy as np
from    collections import deque
from    typing      import Tuple, Deque, Optional
import  random
import  warnings

class DQN(nn.Module):
    """ Init classe rete neurale con 3 livelli nn.Linear """
    def __init__(self, state_dim: int, action_dim: int, device: torch.device, hidden_dim: int = 64) -> None:
        """
        Inizializza un oggetto DQN.
        Parametri:
        - state_dim (int): La dimensione dello stato di input.
        - action_dim (int): La dimensione dell'azione di output.
        - device (torch.device): Il dispositivo su cui eseguire il modello (es. 'cpu', 'cuda').
        - hidden_dim (int): La dimensione dello strato nascosto del modello (default: 64).
        """

        super(DQN, self).__init__()
        self.device = device
        layers = [
            nn.Linear(state_dim, hidden_dim),       # Layer 1 lineare
            nn.ReLU(),                              # Layer 2 Relu per introdurre non linearità
            nn.Linear(hidden_dim, hidden_dim),      # Layer 3 lineare
            nn.ReLU(),                              # Layer 4 Relu per introdurre non linearità
            nn.Linear(hidden_dim, action_dim)       # Layer 5 lineare
        ]
        self.model = nn.Sequential(*layers).to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcola i valori Q attraverso il modello. Questa funazione viene eseguita automaticamente quando si chiama il modello.
        Args:
            x (torch.Tensor): Il tensore di input rappresentante lo stato.
        Returns:
            torch.Tensor: I valori Q calcolati dal modello.
        """

        
        # Calcolo q_values attraverso il modello
        q_values = self.model(x)
        return q_values


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

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        """
        Aggiunge una nuova esperienza < S, A, R, S', terminated > nel buffer.
        Parametri:
        - state: np.ndarray = Lo stato corrente dell'ambiente.
        - action: int = L'azione eseguita nello stato corrente.
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

def select_action(state: np.ndarray, policy_net: torch.nn.Module, epsilon: float, n_actions: int) -> int:
    """
    Seleziona un'azione da prendere in base allo stato corrente utilizzando una rete neurale come policy network.
    Args:
        state (np.ndarray): L'array numpy che rappresenta lo stato corrente.
        policy_net (torch.nn.Module): La rete neurale utilizzata come policy network.
        epsilon (float): Il valore di epsilon che determina la probabilità di selezionare un'azione casuale.
        n_actions (int): Il numero totale di azioni disponibili.
    Returns:
        int: L'indice dell'azione selezionata.
    """
    
    # Converto lo stato in torch.Tensor e lo sposto sul device corretto
    state_tensor = torch.tensor(state, dtype=torch.float32, device = policy_net.device).unsqueeze(0)
    # Azione casuale con probabilità epsilon
    if random.random() < epsilon:
        return random.randint(0, n_actions -1)
        
    # Azione greedy con probabilità (1 - epsilon)
    with torch.no_grad():           # Disabilito il calcolo dei gradienti che non serve in questa fase
        # Seleziono l'azione col valore massimo di Q (non serve squeeze(0) perchè sto usando item())
        return policy_net(state_tensor).argmax(dim=1).item()

   
def optimize_model(policy_net: DQN, target_net: DQN, memory: ReplayBuffer, optimizer: Optimizer, batch_size: int, gamma: float, debug: bool = False) -> Optional[torch.Tensor]:
    """
    Ottimizza il modello di rete neurale utilizzando il metodo DQN (Deep Q-Learning).
    Args:
        policy_net (DQN): La rete neurale principale che viene addestrata.
        target_net (DQN): La rete neurale target utilizzata per calcolare i target Q-values.
        memory (ReplayBuffer): Il buffer di replay che contiene le esperienze passate.
        optimizer (Optimizer): L'ottimizzatore utilizzato per aggiornare i pesi della rete.
        batch_size (int): La dimensione del batch di esperienze da campionare dal buffer.
        gamma (float): Il fattore di sconto per il calcolo dei Q-values.
        debug (bool, optional): Se True, restituisce la perdita calcolata per il debug. Default è False.
    Returns:
        Optional[torch.Tensor]: La perdita calcolata se debug è True, altrimenti None.
    """

    # Genera un warning se non ci sono abbastanza esperienze nel replay buffer e non aggiorna la rete
    if len(memory) < batch_size:
        warnings.warn(f"Il replay buffer contiene solo {len(memory)} esperienze, meno del batch size richiesto ({batch_size}).", 
                      category=UserWarning)
        return None

    # Campiono una serie di batch dal buffer
    states, actions, rewards, next_states, terminateds = memory.sample(batch_size)

    # Converto tutto in tensori e sposta sul device della rete policy_net
    device = policy_net.device
    states_tensor       = torch.tensor(states,      dtype=torch.float32).to(device)
    actions_tensor      = torch.tensor(actions,     dtype=torch.int64).unsqueeze(1).to(device)
    rewards_tensor      = torch.tensor(rewards,     dtype=torch.float32).to(device)
    next_states_tensor  = torch.tensor(next_states, dtype=torch.float32).to(device)
    terminateds_tensor  = torch.tensor(terminateds, dtype=torch.float32).to(device)

    # Calcola i Q-valori predetti dalla rete principale per le azioni eseguite
    # policy_net(states) -> tensore (batch_size, n_actions) con i Q values delle azioni per ogni stato nel batch
    # gather(1, actions) -> seleziona i Q-valori associati alle azioni effettivamente eseguite (actions)
    #                       actions è un tensore di indici delle azioni con dimensione (batch_size, 1)
    # squeeze(1) -> rimuove la dimensione extra (batch_size, 1) -> (batch_size,)
    q_values = policy_net(states_tensor).gather(1, actions_tensor).squeeze(1)

    # Calcolo i q_values massimi previsti dalla rete target per gli stati successivi
    # target_net(next_states) -> fornisce un tensore (batch_size, n_actions)
    # max(1) -> estrae i valori massimi lungo la dim 1, ovvero quelli delle azioni e ottengo una tupla
    # con (valori massimi di Q per ogni batch, azione associata per ogni batch)
    # Di questa tupla estraiamo solo i valori massimi di Q, non ci interessa l'azione
    next_q_values = target_net(next_states_tensor).max(1)[0]
    
    # A questo punto calcolo i target per ognuno dei next_q_values ottenuti
    target_q_values = rewards_tensor + gamma * next_q_values * (1 - terminateds_tensor)

    # Calcolo la perdita tra i valori predetti dalla rete principale e quelli della rete
    # target usando l'errore quadratico medio MSE
    loss = nn.MSELoss()(q_values, target_q_values.detach())
    # loss = nn.SmoothL1Loss()(q_values, target_q_values.detach())

    # Esegue il backpropagation e aggiorna i parametri della rete
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if debug:
        return loss
    else:
        return None

def soft_update(policy_net: DQN, target_net: DQN, tau: float) -> None:
    """
    Aggiorna gradualmente i parametri della rete target utilizzando i parametri della rete di policy.
    Questo metodo esegue un aggiornamento soft dei parametri della rete target, combinando i parametri
    della rete di policy con quelli della rete target in base al valore di tau.
    Args:
        policy_net (DQN): La rete di policy i cui parametri verranno utilizzati per aggiornare la rete target.
        target_net (DQN): La rete target i cui parametri verranno aggiornati.
        tau (float): Il fattore di aggiornamento soft. Un valore di tau vicino a 1.0 significa che la rete target
                     sarà aggiornata più velocemente verso i parametri della rete di policy, mentre un valore
                     vicino a 0.0 significa un aggiornamento più lento.
    """
    
    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Funzione per il decadimento di epsilon
def eps_decay(episode: int, max_episodes: int, curr_eps: float, min_eps: float, max_eps: float, exploration_factor: float, eps_decay: Optional[float], decay_type: str='linear') -> float:
    """
    Calcola il valore epsilon decrescente per un algoritmo di apprendimento per rinforzo.
    Args:
        episode (int): L'episodio corrente.
        max_episodes (int): Il numero massimo di episodi.
        curr_eps (float): Il valore corrente di epsilon.
        min_eps (float): Il valore minimo di epsilon.
        max_eps (float): Il valore massimo di epsilon.
        eps_decay (float, opzionale): Il tasso di decadimento per il tipo di decadimento esponenziale. Default è 0.0.
        decay_type (str, opzionale): Il tipo di decadimento ('linear' o 'exp'). Default è 'linear'.
    Returns:
        float: Il nuovo valore di epsilon dopo il decadimento.
    """
    
    if decay_type == 'linear':
        return np.max([min_eps, max_episodes * (1 - episode / (max_episodes * exploration_factor))])
        
    elif decay_type == 'exp':
        assert eps_decay is not None
        return np.max([min_eps, max_eps * np.exp(- eps_decay * episode)])
    else:
        return 0 