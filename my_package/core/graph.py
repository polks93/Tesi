import numpy as np
import matplotlib.pyplot as plt
from zeno import Zeno

if __name__ == "__main__":
    
    zeno = Zeno()
    surge_des = 0.1
    data = np.load("C:/Users/paolo/Desktop/Riferimenti/v_surge1.npz")
    time = data['time']
    vel = data['vel']
    ref = data['ref']

    N = len(time)
    t0 = 0
    surge_his = []

    for i in range(N):
        t1 = time[i]
        dt = t1 - t0
        zeno.controller(surge_des, 0, 0, dt=dt)
        zeno.kinematics(dt=dt)
        state = zeno.get_state()
        surge = state[3]
        surge_his.append(surge)
        t0 = t1

    fig, axs = plt.subplots(1,3 , figsize=(10, 10))
    axs[0].plot(time, surge_his, label='Vx sim', color='red')
    axs[0].plot(time, vel, label='Vx Zeno', color='green')
    axs[0].plot(time, ref, label='Vx des', linestyle='--', color='black')
    axs[0].grid(True)
    axs[0].legend(fontsize=16)
    axs[0].set_title('Andamento Vx', fontsize=16)
    axs[0].set_xlabel('Tempo [s]', fontsize=16)
    axs[0].set_ylabel('Vx [m/s]', fontsize=16)
    axs[0].set_xlim(0, 25)

    data = np.load("C:/Users/paolo/Desktop/Riferimenti/v_sway1.npz")
    time = data['time']
    vel = - data['vel']
    ref = - data['ref']
    sway_des = 0.1
    N = len(time)
    t0 = 0
    sway_his = []

    for i in range(N):
        t1 = time[i]
        dt = t1 - t0
        zeno.controller(0, sway_des, 0, dt=dt)
        zeno.kinematics(dt=dt)
        state = zeno.get_state()
        sway = state[4]
        sway_his.append(sway)
        t0 = t1




    axs[1].plot(time, sway_his, label='Vy sim', color='red')
    axs[1].plot(time, vel, label='Vy Zeno', color='green')
    axs[1].plot(time, ref, label='Vy des', linestyle='--', color='black')
    axs[1].grid(True)
    axs[1].legend(fontsize=16)
    axs[1].set_title('Andamento Vy', fontsize=16)
    axs[1].set_xlabel('Tempo [s]', fontsize=16)
    axs[1].set_ylabel('Vy [m/s]', fontsize=16)
    axs[1].set_xlim(0, 25)


    data = np.load("C:/Users/paolo/Desktop/Riferimenti/omega45.npz")
    time = data['time']
    vel = - data['vel']
    ref = - data['ref']
    theta_des = np.deg2rad(45)

    N = len(time)
    t0 = 0
    omega_his = []
    for i in range(N):
        theta = zeno.get_state()[2]
        theta_rel = theta_des - theta
        t1 = time[i]
        dt = t1 - t0
        zeno.controller(0, 0, theta_rel, dt=dt)
        zeno.kinematics(dt=dt)
        state = zeno.get_state()
        theta = np.rad2deg(state[2])

        omega = state[5]
        omega_his.append(omega)
        t0 = t1
    
    axs[2].plot(time, omega_his, label='Omega sim', color='red')
    axs[2].plot(time, vel, label='Omega Zeno', color='green')
    axs[2].grid(True)
    axs[2].legend(fontsize=16)
    axs[2].set_title('Andamento Omega (theta rel = 45°)', fontsize=16) 
    axs[2].set_xlabel('Tempo [s]', fontsize=16)
    axs[2].set_ylabel('Omega [rad/s]', fontsize=16)
    axs[2].set_xlim(0, 25)


    # Adjust tick label font sizes for all subplots
    for ax in axs:
        ax.tick_params(axis='both', labelsize=16)  # Set fontsize for tick labels

    plt.show()

    