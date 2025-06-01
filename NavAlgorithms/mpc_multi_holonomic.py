import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Simulation parameters
N = 5          # prediction horizon
dt = 0.1        # time step
num_bots = 4
# Cost function weights
Q = np.diag([10.0, 10.0, 1.0])  # penalize x, y, theta errors
R = np.diag([2.0, 2.0, 2.0])         # penalize control effort [v, omega]
W_dir = 1.0
W_collision= 0.1

initial_poses = np.array([[0.4, 0.4, 0.0], [0.8, 0.5, 0.0], [1.0, 0.0, 0.0], [0.25,0.8,0.0]])
final_poses = np.array([[1.0, 1.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0,0.5,0.0]]) # final pose after 10 iterations

# Reference trajectory (straight line along x-axis)
x_ref = np.zeros((num_bots, N))
y_ref = np.zeros((num_bots, N))
theta_ref = np.zeros((num_bots, N))
ref_traj = np.zeros((num_bots,N, 3))

for bot in range(num_bots):
    x_ref[bot] = np.linspace(initial_poses[bot][0], final_poses[bot][0], N)
    y_ref[bot] = np.linspace(initial_poses[bot][1], final_poses[bot][1], N)
    theta_ref[bot] = np.zeros(N)
    ref_traj[bot] = np.stack([x_ref[bot], y_ref[bot], theta_ref[bot]], axis=1) 

    # print(ref_traj[bot])

# Differential drive motion model
def holonomoic_drive_model(x, u):
    x_new = np.zeros_like(x)
    vx, vy, omega = u
    x_new[0] = x[0] + vx * dt
    x_new[1] = x[1] + vy * dt
    x_new[2] = x[2] + omega * dt
    return x_new

def objective(U_flat):
    U = U_flat.reshape(num_bots, N, 3)  # [vx, vy, omega]
    cost = 0.0

    # Store simulated states for each robot over the horizon
    x_all = [initial_poses[bot].copy() for bot in range(num_bots)]
    prev_theta = [np.arctan2(U[bot][0][1], U[bot][0][0]) for bot in range(num_bots)]

    for t in range(N):
        # Step all robots
        for bot in range(num_bots):
            x = holonomoic_drive_model(x_all[bot], U[bot][t])

            x_all[bot] = x
            # 1. Tracking error
            x_err = x - ref_traj[bot][t]

            # 2. Control effort
            u_err = U[bot][t]

            # 3. Direction smoothness
            vx, vy = U[bot][t][0], U[bot][t][1]
            theta = np.arctan2(vy, vx)
            dir_error = (theta - prev_theta[bot])**2
            prev_theta[bot] = theta

            # Accumulate cost
            cost += x_err.T @ Q @ x_err + u_err.T @ R @ u_err + W_dir * dir_error

        # 4. Inter-robot collision avoidance cost
        for i in range(num_bots):
            for j in range(i + 1, num_bots):
                xi = x_all[i][:2]
                xj = x_all[j][:2]
                dist = np.linalg.norm(xi - xj)
                # print("dist  ", W_collision / (dist**2 + 1e-3))
                # print("cost  ", cost)
                cost += W_collision / (dist**2 + 1e-3)  # Add epsilon to avoid div by zero

    return cost

# Run optimizer
U0 = np.random.rand(num_bots, N, 3)
U0[:, :, -1] = 0.0  # omega = 0 for now
result = minimize(objective, U0.flatten(), method='SLSQP')
U_opt = result.x.reshape(num_bots, N, 3)

# Simulate with optimal control sequence
x_all = [initial_poses[bot].copy() for bot in range(num_bots)]
trajectory = [[] for _ in range(num_bots)]  # list of list of poses

for bot in range(num_bots):
    x = x_all[bot].copy()
    trajectory[bot].append(x.copy())  # starting point

    for t in range(N):
        x = holonomoic_drive_model(x, U_opt[bot][t])
        trajectory[bot].append(x.copy())

# Convert to numpy arrays
trajectory = [np.array(traj) for traj in trajectory]

# --- Plotting ---
colors = ['b', 'r', 'm', 'c', 'orange', 'purple']  # Customize as needed
plt.figure(figsize=(10, 6))

for bot in range(num_bots):
    traj = trajectory[bot]
    plt.plot(traj[:, 0], traj[:, 1], color=colors[bot % len(colors)], label=f'Bot {bot} MPC Path')
    plt.plot(ref_traj[bot][:, 0], ref_traj[bot][:, 1], '--', color=colors[bot % len(colors)], alpha=0.5, label=f'Bot {bot} Ref')
    plt.scatter(initial_poses[bot][0], initial_poses[bot][1], marker='o', c=colors[bot % len(colors)])

plt.xlabel('x')
plt.ylabel('y')
plt.title('MPC Trajectories for Multiple Holonomic Robots')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()