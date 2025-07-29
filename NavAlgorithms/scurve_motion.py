import numpy as np
import matplotlib.pyplot as plt

def generate_s_curve_profile(V_max, A_max, total_time, dt):
    # Assume jerk is maximum acceleration in 1/3rd of acceleration phase
    Tj = 0.2  # jerk phase duration (can also be input)
    J_max = A_max / Tj  # max jerk
    
    Ta = 2 * Tj + (V_max / A_max - Tj)  # acceleration duration
    Td = Ta  # symmetric deceleration
    Tc = total_time - Ta - Td  # constant velocity phase duration

    if Tc < 0:
        raise ValueError("Total time too short for given V_max and A_max to fit S-curve")

    print(f"Jerk: {J_max:.2f}, Accel Time: {Ta:.2f}s, Cruise Time: {Tc:.2f}s, Decel Time: {Td:.2f}s")

    time = np.arange(0, total_time, dt)
    pos, vel, acc = [], [], []

    for t in time:
        if t < Tj:
            # Phase 1: Jerk up
            a = J_max * t
            v = 0.5 * J_max * t**2
            p = (1/6) * J_max * t**3
        elif t < (Ta - Tj):
            # Phase 2: Constant acceleration
            t1 = t - Tj
            a = A_max
            v = 0.5 * J_max * Tj**2 + A_max * t1 # u + a.t
            p = ((1/6) * J_max * Tj**3) + 0.5 * A_max * t1**2 + (0.5 * J_max * Tj**2) * t1 # xt + 0.5at^2 + ut
        elif t < Ta:
            # Phase 3: Jerk down
            t2 = t - (Ta - Tj)
            a = A_max - J_max * t2
            v = (V_max - 0.5 * J_max * t2**2)
            p = (V_max * t2 - (1/6) * J_max * t2**3) + (V_max - A_max * t2) * (Ta - Tj - t2) # (x + v.t)
        elif t < Ta + Tc:
            # Phase 4: Constant velocity
            t3 = t - Ta
            a = 0
            v = V_max
            p = (V_max * t3) + position_at_Ta(V_max, A_max, Tj)
        elif t < Ta + Tc + Tj:
            # Phase 5: Jerk down (deceleration)
            t4 = t - (Ta + Tc)
            a = -J_max * t4
            v = V_max - 0.5 * J_max * t4**2
            p = position_at_TaTc(V_max, A_max, Tj, Tc) + V_max * t4 - (1/6) * J_max * t4**3
        elif t < total_time - Tj:
            # Phase 6: Constant deceleration
            t5 = t - (Ta + Tc + Tj)
            a = -A_max
            v = V_max - 0.5 * J_max * Tj**2 - A_max * t5
            p = position_at_TaTc(V_max, A_max, Tj, Tc) + area_const_decel(t5, A_max, J_max, Tj)
        else:
            # Phase 7: Jerk up (to 0 accel)
            t6 = t - (total_time - Tj)
            a = -A_max + J_max * t6
            v = 0.5 * J_max * (Tj - t6)**2
            p = position_at_end(V_max, A_max, Tj, Tc) + area_final_jerk(t6, J_max, Tj)
        
        pos.append(p)
        vel.append(v)
        acc.append(a)

    return time, np.array(pos), np.array(vel), np.array(acc)

# Helper functions
def position_at_Ta(V_max, A_max, Tj):
    return (1/6)*A_max*Tj**2*(3*(V_max/A_max - Tj) + Tj)

def position_at_TaTc(V_max, A_max, Tj, Tc):
    return position_at_Ta(V_max, A_max, Tj) + V_max * Tc

def area_const_decel(t, A_max, J_max, Tj):
    return (0.5 * A_max * t**2 + 0.5 * J_max * Tj**2 * t)

def position_at_end(V_max, A_max, Tj, Tc):
    return position_at_TaTc(V_max, A_max, Tj, Tc) + 0.5 * A_max * Tj**2

def area_final_jerk(t, J_max, Tj):
    return (0.5 * J_max * (Tj - t)**2 * t)

# Run and plot
if __name__ == "__main__":
    V_max = 1.0        # m/s
    A_max = 2.0        # m/s^2
    total_time = 5.0   # seconds
    dt = 0.01          # time resolution

    t, p, v, a = generate_s_curve_profile(V_max, A_max, total_time, dt)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, p)
    plt.ylabel('Position (m)')

    plt.subplot(3, 1, 2)
    plt.plot(t, v)
    plt.ylabel('Velocity (m/s)')

    plt.subplot(3, 1, 3)
    plt.plot(t, a)
    plt.ylabel('Acceleration (m/s²)')
    plt.xlabel('Time (s)')

    plt.tight_layout()
    plt.show()
