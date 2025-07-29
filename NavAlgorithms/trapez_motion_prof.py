import numpy as np
import matplotlib.pyplot as plt

def trapezoid_profile(target_position, total_time, dt):
    # Parameters
    A_max = 1.54 # rad/s^2 it can reach to 1.57 radians in 1 second 

    # Calculate max velocity for given target and time
    T_acc = total_time / 2.0
    V_max = target_position / T_acc

    if V_max > A_max * T_acc:
        V_max = A_max * T_acc

    T_phase_1 = V_max / A_max
    T_phase_3 = T_phase_1
    T_phase_2 = total_time - T_phase_1 - T_phase_3

    if T_phase_2 < 0:
        # Triangular profile
        T_phase_1 = np.sqrt(target_position / A_max)
        T_phase_3 = T_phase_1
        T_phase_2 = 0
        V_max = A_max * T_phase_1
        total_time = 2 * T_phase_1

    time = np.arange(0, total_time, dt)
    pos, vel, acc = [], [], []

    v = 0
    s = 0
    print('VEL MAX ', V_max, 'TACC ', T_phase_1, 'TCONV ', total_time - 2 * T_phase_1)
    for t in time:
        if t < T_phase_1:
            a = A_max
            v += a * dt
            s += v * dt
        elif T_phase_1 <= t < T_phase_1 + T_phase_2:
            a = 0
            v = V_max
            s += v * dt
        elif T_phase_1 + T_phase_2 <= t < total_time:
            a = -A_max
            v += a * dt
            if v < 0: v = 0
            s += v * dt
        pos.append(s)
        vel.append(v)
        acc.append(a)

    # Clamp final position to target
    pos = np.array(pos)
    pos = pos - (pos[-1] - target_position)

    return time, pos, np.array(vel), np.array(acc)

# Run and plot
if __name__ == "__main__":
    target_position = 3.00  # meters
    total_time = 5.0      # seconds
    dt = 0.001               # time resolution

    t, p, v, a = trapezoid_profile(target_position, total_time, dt)
    # print(p)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, p)
    plt.ylabel('Position (m)')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t, v)
    plt.ylabel('Velocity (m/s)')
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t, a)
    plt.ylabel('Acceleration (m/s²)')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.tight_layout()
    plt.show()