#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

animate = False
move = True

# Parameters
dt = 0.3
num_bots = 4

# Cost weights
Q = np.diag([20.0, 20.0])
R = np.diag([2.0, 2.0])
W_dir = 0.1
W_collision = 1.0
W_obstacle = 1.0

# Initial and goal positions
initial_poses = np.array([[1.0, 2.0], [2.0, 2.0], [0.0, 2.0], [3.0, 2.0]])
final_poses = np.array([[4.0, 1.0], [0.0, 1.0], [3.0, 0.0], [2.0, 2.0]])

# Obstacles
obs_x = np.array([2] * 10)
obs_y = np.linspace(0.0, 1.5, 10)

# Reference trajectory (only one step)
ref_traj = final_poses.copy()

# Velocity bounds
v_min, v_max = -0.1, 0.1
bounds = [(v_min, v_max), (v_min, v_max)] * num_bots

# Holonomic model
def holonomic_drive_model(x, u):
    return x + u * dt

# Objective for one-step MPC
def objective(U_flat):
    U = U_flat.reshape(num_bots, 2)
    cost = 0.0
    thres = 0.5
    x_all = initial_poses.copy()

    for bot in range(num_bots):
        x = holonomic_drive_model(x_all[bot], U[bot])
        x_err = x - ref_traj[bot]
        u_err = U[bot]
        theta = np.arctan2(U[bot][1], U[bot][0])
        dir_error = theta ** 2  # angle deviation from 0 direction

        cost += x_err.T @ Q @ x_err + u_err.T @ R @ u_err + W_dir * dir_error
        x_all[bot] = x

    # Inter-robot collision
    for i in range(num_bots):
        for j in range(i + 1, num_bots):
            dist = np.linalg.norm(x_all[i] - x_all[j])
            if dist < thres:
                cost += W_collision / (dist**2 + 1e-3)

    # Robot-obstacle collision
    for bot in range(num_bots):
        for obs in zip(obs_x, obs_y):
            obsdist = np.linalg.norm(x_all[bot] - np.array(obs))
            if obsdist < thres:
                cost += W_obstacle / (obsdist + 1e-3)
    print('COST ', cost)

    return cost

# Initial guess
U0 = (final_poses - initial_poses) / dt
U0 = np.clip(U0, v_min, v_max)

# Solve optimization
result = minimize(objective, U0.flatten(), method='SLSQP', bounds=bounds)
U_opt = result.x.reshape(num_bots, 2)

# Visualize motion
if move:
    plt.ion()
    while True:
        trajectories = [[] for _ in range(num_bots)]

        for bot in range(num_bots):
            x = initial_poses[bot].copy()
            trajectories[bot].append(x.copy())
            x = holonomic_drive_model(x, U_opt[bot])
            initial_poses[bot] = x.copy()
            trajectories[bot].append(x.copy())

        trajectories = [np.array(traj) for traj in trajectories]

        plt.clf()
        colors = ['b', 'r', 'm', 'c']
        for bot in range(num_bots):
            traj = trajectories[bot]
            plt.plot(traj[:, 0], traj[:, 1], color=colors[bot], label=f'Bot {bot}')
            plt.scatter(initial_poses[bot][0], initial_poses[bot][1], marker='o', c=colors[bot], s=100)

        plt.scatter(obs_x, obs_y, c='k', s=100, marker='o', label='Obstacles')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.pause(0.1)

        # Recalculate U_opt based on updated initial positions
        U0 = (final_poses - initial_poses) / dt
        U0 = np.clip(U0, v_min, v_max)
        result = minimize(objective, U0.flatten(), method='SLSQP', bounds=bounds)
        U_opt = result.x.reshape(num_bots, 2)
