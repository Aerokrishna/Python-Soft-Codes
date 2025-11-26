import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from matplotlib.animation import FuncAnimation

#  Quarter-Car Model Parameters
m_s = 250.0     # sprung mass (kg)
m_u = 40.0      # unsprung mass (kg)
k_s = 15000.0   # suspension stiffness (N/m)
k_t = 200000.0  # tire stiffness (N/m)
b_s = 1000.0    # passive damping (N·s/m)

#  State-space matrices
A = np.array([
    [0, 1, 0, 0],
    [-k_s/m_s, -b_s/m_s, k_s/m_s, b_s/m_s],
    [0, 0, 0, 1],
    [k_s/m_u, b_s/m_u, -(k_s + k_t)/m_u, -b_s/m_u]
])
B = np.array([[0], [1/m_s], [0], [-1/m_u]])
E = np.array([[0], [0], [0], [k_t/m_u]])

#  LQR Gain Computation
Q = np.diag([10000, 1000, 1000, 10])   # penalize z_s, zsdot, z_u, zudot
R = np.array([[1]])                    # penalize actuator force effort

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ (B.T @ P)

print("LQR Gain K:", K)

#  Simulation Parameters
dt = 0.001       # time step (s)
T_end = 5.0
N = int(T_end / dt)
t = np.linspace(0, T_end, N)

# -------------------------------
# Road Profile Definitions
# -------------------------------

# Existing: half-sine bump at t = 1.0s
t_start = 1.0
t_dur = 0.1
bump_amp = 0.05

def road_profile_step(time):
    """Half-sine bump"""
    if t_start <= time <= t_start + t_dur:
        tau = (time - t_start) / t_dur
        return bump_amp * np.sin(np.pi * tau)
    else:
        return 0.0

# NEW 1️⃣: SINE WAVE ROAD (continuous uneven road)
def road_profile_sine(time):
    freq = 2.0        # Hz
    amplitude = 0.03  # meters
    return amplitude * np.sin(2 * np.pi * freq * time)

# NEW 2️⃣: RAMP ROAD (gradually rising road level)
def road_profile_ramp(time):
    slope = 0.02   # meters per second
    t_ramp_end = 2.0
    if time <= t_ramp_end:
        return slope * time
    else:
        return slope * t_ramp_end

road_profile = road_profile_step        # original
road_profile = road_profile_sine      # sine road
# road_profile = road_profile_ramp      # ramp road
# --------------------------------------


#  Dynamics simulation (Euler or RK4)
def dynamics(x, u, zr):
    dx = A @ x + B * u + E * zr
    return dx

def rk4_step(x, u, zr, dt):
    k1 = dynamics(x, u, zr)
    k2 = dynamics(x + 0.5*dt*k1, u, zr)
    k3 = dynamics(x + 0.5*dt*k2, u, zr)
    k4 = dynamics(x + dt*k3, u, zr)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# Initial conditions
x = np.zeros((4, 1))

# Logging
zs, zu, fa_log, acc_s = [], [], [], []
zr_log = []

for ti in t:
    zr = road_profile(ti)
    u = -K @ x               # LQR control
    x = rk4_step(x, u, zr, dt)

    # Log data
    zs.append(x[0,0])
    zu.append(x[2,0])
    zr_log.append(zr)
    fa_log.append(u[0,0])
    acc_s.append((-k_s*(x[0]-x[2]) - b_s*(x[1]-x[3]) + u[0]) / m_s)

zs, zu, zr_log, fa_log, acc_s = map(np.array, [zs, zu, zr_log, fa_log, acc_s])

#  Static Plots
plt.figure(figsize=(10, 3))
plt.plot(t, zs, label='Sprung (Body)')
plt.plot(t, zu, label='Unsprung (Wheel)')
plt.plot(t, zr_log, label='Road')
plt.title("Displacements (LQR)")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 3))
plt.plot(t, zs - zu)
plt.title("Suspension Deflection (zs - zu) (LQR)")
plt.xlabel("Time [s]")
plt.grid(True)

plt.figure(figsize=(10, 3))
plt.plot(t, acc_s)
plt.title("Body Acceleration (m/s²) (LQR)")
plt.xlabel("Time [s]")
plt.grid(True)

plt.figure(figsize=(10, 3))
plt.plot(t, fa_log)
plt.title("Actuator Force (N) (LQR)")
plt.xlabel("Time [s]")
plt.grid(True)

plt.show()

enable_live_plot = True
if enable_live_plot:
    fig, ax = plt.subplots(figsize=(8,3))
    line1, = ax.plot([], [], lw=2, label="Body (z_s)")
    line2, = ax.plot([], [], lw=2, label="Wheel (z_u)")
    line3, = ax.plot([], [], lw=1, ls='--', label="Road (z_r)")
    ax.set_xlim(0, T_end)
    ax.set_ylim(-0.1, 0.1)
    ax.set_title("Live LQR Suspension Simulation")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [m]")
    ax.grid(True)
    ax.legend()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3

    def update(frame):
        idx = frame
        line1.set_data(t[:idx], zs[:idx])
        line2.set_data(t[:idx], zu[:idx])
        line3.set_data(t[:idx], zr_log[:idx])
        return line1, line2, line3

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init,
                        blit=True, interval=1)
    plt.show()
