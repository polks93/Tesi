import numpy as np
import os
import matplotlib.pyplot as plt
from RL import eps_decay

def smooth(data, window_size):
    # """Applica una media mobile a un array."""
    # window = np.ones(window_size) / window_size
    # return np.convolve(data, window, mode='valid')
    smoothed_data = np.zeros(len(data))  # Inizializza l'array per i risultati
    
    for i in range(len(data)):
        # Calcola il numero di elementi da considerare (finestra fino all'inizio)
        start_idx = max(0, i - window_size + 1)
        smoothed_data[i] = np.mean(data[start_idx:i + 1])
    
    return smoothed_data

def fill_with_last_nonzero(data):
    # """Riempi i valori nulli con l'ultimo valore non nullo."""
    last_nonzero = 0
    first_nonzero_idx = 0

    for i in range(len(data)):
        if data[i] != 0:
            first_nonzero_idx = i
            break


    for i in range(first_nonzero_idx, len(data)):
        if data[i] == 0:
            data[i] = last_nonzero
        else:
            last_nonzero = data[i]

    return data
if __name__ == '__main__':

    directory1 = 'C:/Users/paolo/Documents/Università/JN/SHIP/ShipQuest-v4_Data/multi'
    # directory1 = 'C:/Users/paolo/Documents/Università/JN/DDPG/multi'

    # directory2 = 'C:/Users/paolo/Documents/Università/JN/SHIP/ShipQuest-v1_Data/multi_no_prox'

    N_sim = 2
    sim_data1 = []
    # sim_data2 = []
    for i in range(N_sim):
        file_name = 'sim' + str(i) + '.npz'
        file_path = os.path.join(directory1, file_name)
        data = np.load(file_path)
        sim_data1.append(data)
        # file_path = os.path.join(directory2, file_name)
        # data = np.load(file_path)
        # sim_data2.append(data)


    # one_len_ep = sim_data[0]['len_ep']
    len_ep = [data['len_ep'] for data in sim_data1]
    reward = [data['reward'] for data in sim_data1]
    coverage = [data['coverage'] for data in sim_data1]
    trained_coverage_grouped = [data['trained_coverage'] for data in sim_data1]
    
    # len_ep2 = [data['len_ep'] for data in sim_data2]
    # reward2 = [data['reward'] for data in sim_data2]
    # coverage2 = [data['coverage'] for data in sim_data2]
    # trained_coverage_grouped2 = [data['trained_coverage'] for data in sim_data2]


    trained_coverage_raw = np.concatenate(trained_coverage_grouped, axis=0)
    # trained_coverage_raw2 = np.concatenate(trained_coverage_grouped2, axis=0)


    trained_coverage = []
    # trained_coverage2 = []

    for i in range(len(trained_coverage_raw)):
        trained_coverage.append(fill_with_last_nonzero(trained_coverage_raw[i]))

    # for i in range(len(trained_coverage_raw2)):
    #     trained_coverage2.append(fill_with_last_nonzero(trained_coverage_raw2[i]))


    window_size = 25
    len_ep_mean = smooth(np.mean(len_ep, axis=0), window_size)
    len_ep_std = smooth(np.std(len_ep, axis=0), window_size)
    reward_mean = smooth(np.mean(reward, axis=0), window_size)
    reward_std = smooth(np.std(reward, axis=0), window_size)
    coverage_mean = smooth(np.mean(coverage, axis=0), window_size)
    coverage_std = smooth(np.std(coverage, axis=0), window_size)
    trained_coverage_mean = np.mean(trained_coverage, axis=0)
    trained_coverage_std = np.std(trained_coverage, axis=0)

    # len_ep_mean2 = smooth(np.mean(len_ep2, axis=0), window_size)
    # len_ep_std2 = smooth(np.std(len_ep2, axis=0), window_size)
    # reward_mean2 = smooth(np.mean(reward2, axis=0), window_size)
    # reward_std2 = smooth(np.std(reward2, axis=0), window_size)
    # coverage_mean2 = smooth(np.mean(coverage2, axis=0), window_size)
    # coverage_std2 = smooth(np.std(coverage2, axis=0), window_size)
    # trained_coverage_mean2 = np.mean(trained_coverage2, axis=0)
    # trained_coverage_std2 = np.std(trained_coverage2, axis=0)


    plt.figure(figsize=(10, 6))
    # plt.plot(trained_coverage_mean2, label='Metodo 1 DQN')
    # plt.fill_between(np.arange(len(trained_coverage_mean2)), trained_coverage_mean2 - trained_coverage_std2, trained_coverage_mean2 + trained_coverage_std2, alpha=0.5)
    plt.plot(trained_coverage_mean)
    plt.fill_between(np.arange(len(trained_coverage_mean)), trained_coverage_mean - trained_coverage_std, trained_coverage_mean + trained_coverage_std, alpha=0.5)
    plt.title('Copertura del perimetro agente addestrato (Metodo DDPG)', fontsize=14)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Copertura media', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 6))
    # plt.plot(len_ep_mean2, label='Metodo 1 DQN')
    # plt.fill_between(np.arange(len(len_ep_mean2)), len_ep_mean2 - len_ep_std2, len_ep_mean2 + len_ep_std2, alpha=0.5)
    plt.plot(len_ep_mean)
    plt.fill_between(np.arange(len(len_ep_mean)), len_ep_mean - len_ep_std, len_ep_mean + len_ep_std, alpha=0.5)
    plt.title('Lunghezza episodi addestramento (Metodo DDPG)', fontsize=14)
    plt.xlabel('Episodi di addestramento', fontsize=14)
    plt.ylabel('Lunghezza media', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    # plt.plot(reward_mean2, label='Metodo 1 DQN')
    # plt.fill_between(np.arange(len(reward_mean2)), reward_mean2 - reward_std2, reward_mean2 + reward_std2, alpha=0.5)
    plt.plot(reward_mean)
    plt.fill_between(np.arange(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.5)
    plt.title("Ricompense durante l'addestramento (Metodo DDPG)", fontsize=14)
    plt.xlabel('Episodi di addestramento', fontsize=14)
    plt.ylabel('Ricompensa media', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("Copertura del perimentro durante l'addestramento (Metodo DDPG)", fontsize=14)
    # plt.plot(coverage_mean2, label='Metodo 1 DQN')
    # plt.fill_between(np.arange(len(coverage_mean2)), coverage_mean2 - coverage_std2, coverage_mean2 + coverage_std2, alpha=0.5)
    plt.plot(coverage_mean)
    plt.fill_between(np.arange(len(coverage_mean)), coverage_mean - coverage_std, coverage_mean + coverage_std, alpha=0.5)
    plt.xlabel('Episodi di addestramento', fontsize=14)
    plt.ylabel('Copertura media', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()





