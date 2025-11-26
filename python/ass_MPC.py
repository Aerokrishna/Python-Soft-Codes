import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#  Quarter-Car 
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

# MPC params
dt = 0.001       # time step (s)
T_end = 5.0
N = int(T_end / dt)
t = np.linspace(0, T_end, N)

# weights
Qx = np.diag([50000, 20000, 2000, 1000])  # state weight per step
R = np.array([[0.001]])                  # input weight per step

# prediction horizon (number of discrete steps)
Hp = 50   # 50 steps -> 0.05 s horizon (fast enough for suspension)
eps_reg = 1e-6  # small regularization for numerical stability

# discrete-time approximation 
n = A.shape[0]
m = B.shape[1]
A_d = np.eye(n) + A * dt
B_d = B * dt
E_d = E * dt

# Build prediction matrices (Phi, Gamma, E_bar)
# Phi: (n*Hp x n) ; Gamma: (n*Hp x m*Hp) ; E_bar: (n*Hp x Hp)
Phi = np.zeros((n * Hp, n))
Gamma = np.zeros((n * Hp, m * Hp))
E_bar = np.zeros((n * Hp, Hp))

# Precompute powers of A_d
A_pows = [np.eye(n)]
for i in range(1, Hp + 1):
    A_pows.append(A_pows[-1] @ A_d)  # A_d^i

for i in range(Hp):
    # Phi block row
    Phi[i*n:(i+1)*n, :] = A_pows[i+1]  # A_d^(i+1)
    # Gamma block rows
    for j in range(i+1):
        # contribution of u_j to x_{i+1} is A_d^(i-j) * B_d
        Gamma[i*n:(i+1)*n, j*m:(j+1)*m] = A_pows[i-j] @ B_d
        # contribution of zr_j to x_{i+1} is A_d^(i-j) * E_d
        E_bar[i*n:(i+1)*n, j] = (A_pows[i-j] @ E_d).flatten()

# Block-diagonal cost matrices
Qbar = np.kron(np.eye(Hp), Qx)   # (n*Hp x n*Hp)
Rbar = np.kron(np.eye(Hp), R)    # (m*Hp x m*Hp)

# reference state (we want zero body/wheel deflection/velocities)
x_ref_flat = np.zeros((n * Hp, 1))

# Road Profile
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

def road_profile_sine(time):
    freq = 2.0        # Hz
    amplitude = 0.03  # meters
    return amplitude * np.sin(2 * np.pi * freq * time)

def road_profile_ramp(time):
    slope = 0.02   # meters per second
    t_ramp_end = 2.0
    if time <= t_ramp_end:
        return slope * time
    else:
        return slope * t_ramp_end

# choose profile
# road_profile = road_profile_step
road_profile = road_profile_sine
# road_profile = road_profile_ramp


# Simulation (MPC in the loop)
def dynamics(x, u, zr):
    dx = A @ x + B * u + E * zr
    return dx

def rk4_step(x, u, zr, dt):
    k1 = dynamics(x, u, zr)
    k2 = dynamics(x + 0.5*dt*k1, u, zr)
    k3 = dynamics(x + 0.5*dt*k2, u, zr)
    k4 = dynamics(x + dt*k3, u, zr)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# initial state
x = np.zeros((n, 1))

zs, zu, fa_log, acc_s = [], [], [], []
zr_log = []

# Precompute H (Hessian) which is constant: H = 2*(Gamma^T Qbar Gamma + Rbar)
H_const = 2 * (Gamma.T @ Qbar @ Gamma + Rbar)
# add small regularization to diagonal for numeric stability
H_const += eps_reg * np.eye(H_const.shape[0])

# Main loop
for k_idx, ti in enumerate(t):
    # get current measured road zr_k
    zr_k = road_profile(ti)
    
    # build predicted future road vector zr_pred of length Hp
    zr_pred = np.zeros((Hp, 1))
    for j in range(Hp):
        tz = ti + j * dt
        zr_pred[j, 0] = road_profile(tz)

    # x_free = Phi x_k + E_bar * zr_pred
    x_free = Phi @ x + E_bar @ zr_pred  # (n*Hp x 1)

    # compute f = 2 * Gamma^T * Qbar * (x_free - x_ref_flat)
    f = 2 * (Gamma.T @ Qbar @ (x_free - x_ref_flat))  # (m*Hp x 1)

    # solve quadratic problem H U + f = 0  ->  U = - H^{-1} f
    U_opt = -np.linalg.solve(H_const, f).reshape(m * Hp, 1)

    # apply only first control input (receding horizon)
    u = U_opt[0:m].reshape(m, 1)

    # step dynamics (RK4) using continuous model
    x = rk4_step(x, u, zr_k, dt)

    # Logging
    zs.append(x[0,0])
    zu.append(x[2,0])
    zr_log.append(zr_k)
    fa_log.append(float(u[0,0]))
    acc_s.append((-k_s*(x[0]-x[2]) - b_s*(x[1]-x[3]) + u[0]) / m_s)

# convert logs to arrays
zs, zu, zr_log, fa_log, acc_s = map(np.array, [zs, zu, zr_log, fa_log, acc_s])

#  Static Plots
plt.figure(figsize=(10, 3))
plt.plot(t, zs, label='Sprung (Body)')
plt.plot(t, zu, label='Unsprung (Wheel)')
plt.plot(t, zr_log, label='Road')
plt.title("Displacements (MPC)")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 3))
plt.plot(t, zs - zu)
plt.title("Suspension Deflection (zs - zu) (MPC)")
plt.xlabel("Time [s]")
plt.grid(True)

plt.figure(figsize=(10, 3))
plt.plot(t, acc_s)
plt.title("Body Acceleration (m/s²) (MPC)")
plt.xlabel("Time [s]")
plt.grid(True)

plt.figure(figsize=(10, 3))
plt.plot(t, fa_log)
plt.title("Actuator Force (N) (MPC)")
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
    ax.set_title("Live MPC Suspension Simulation")
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
