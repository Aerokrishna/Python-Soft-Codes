#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

animate = False
move = True
# Simulation parameters
N = 5          # prediction horizon
dt = 0.1        # time step
num_bots = 4

# Cost function weights
Q = np.diag([20.0, 20.0])  # penalize x, y, theta errors
R = np.diag([2.0, 2.0])         
W_dir = 0.0
W_collision= 0.1

# initial_poses = np.array([[1.0, 1.0], [3.0, 2.0], [1.0, 3.0]])
# final_poses = np.array([[3.0, 3.0], [1.0, 2.0], [3.0, 1.0]]) # final pose after 10 iterations

initial_poses = np.random.uniform(0.5, 3.0, size=(num_bots, 2))
final_poses = np.random.uniform(0.5, 3.0, size=(num_bots, 2))

# Reference trajectory (straight line along x-axis)
x_ref = np.zeros((num_bots, N))
y_ref = np.zeros((num_bots, N))
ref_traj = np.zeros((num_bots,N, 2))

v_min, v_max = -0.2, 0.2     # min and max linear velocity in x and y
bounds = []

for _ in range(num_bots):
    for _ in range(N):
        bounds.append((v_min, v_max))        # vx
        bounds.append((v_min, v_max))        # vy

for bot in range(num_bots):
    x_ref[bot] = np.linspace(initial_poses[bot][0], final_poses[bot][0], N)
    y_ref[bot] = np.linspace(initial_poses[bot][1], final_poses[bot][1], N)
    ref_traj[bot] = np.stack([x_ref[bot], y_ref[bot]], axis=1) 

    # print(ref_traj[bot])

# Differential drive motion model
def holonomoic_drive_model(x, u):
    x_new = np.zeros_like(x)
    vx, vy = u
    x_new[0] = x[0] + vx * dt
    x_new[1] = x[1] + vy * dt
    # x_new[2] = x[2] + omega * dt
    return x_new

def objective(U_flat):
    U = U_flat.reshape(num_bots, N, 2)  # [vx, vy]
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
                xi = x_all[i]
                xj = x_all[j]
                dist = np.linalg.norm(xi - xj)
                # print("dist  ", W_collision / (dist**2 + 1e-3))
                # print("cost  ", cost)
                # cost += W_collision / (dist**2 + 1e-3)  # Add epsilon to avoid div by zero
                thres = 0.5
                if dist < thres :
                    cost += W_collision / (dist**2 + 1e-3)  # Add epsilon to avoid div by zero

                    # cost = W_collision * np.exp(-1.0 * dist**2)

    return cost

# Run optimizer
U0 = np.zeros((num_bots, N, 2))
for bot in range(num_bots):
    vx = (final_poses[bot][0] - initial_poses[bot][0]) / (N * dt)
    vy = (final_poses[bot][1] - initial_poses[bot][1]) / (N * dt)
    U0[bot, :, 0] = vx
    U0[bot, :, 1] = vy
    
print(len(U0.flatten()), "  ", len(bounds))
result = minimize(objective, U0.flatten(), method='SLSQP', bounds=bounds)
U_opt = result.x.reshape(num_bots, N, 2)
# print("bot 1 control : ", U_opt[0][0], "bot 2 control : ", U_opt[1][0], "bot 3 control : ", U_opt[2][0], "bot 4 control : ", U_opt[3][0])

if move:
    plt.ion()  # turn on interactive mode
    while True:
        x_all = [initial_poses[bot].copy() for bot in range(num_bots)]
        trajectory = [[] for _ in range(num_bots)]

        for bot in range(num_bots):
            x = x_all[bot].copy()
            trajectory[bot].append(x.copy())

            for t in range(N):
                x = holonomoic_drive_model(x, U_opt[bot][t])
                trajectory[bot].append(x.copy())

        trajectory = [np.array(traj) for traj in trajectory]

        # Plot
        plt.clf()
        colors = ['b', 'r', 'm', 'c', 'orange', 'purple']
        for bot in range(num_bots):
            traj = trajectory[bot]
            plt.plot(traj[:, 0], traj[:, 1], color=colors[bot % len(colors)], label=f'Bot {bot} MPC Path')
            plt.plot(ref_traj[bot][:, 0], ref_traj[bot][:, 1], '--', color=colors[bot % len(colors)], alpha=0.5)
            plt.scatter(initial_poses[bot][0], initial_poses[bot][1], marker='o', c=colors[bot % len(colors)], s=100)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('MPC Trajectories for Multiple Holonomic Robots')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.pause(0.1)  # Non-blocking plot

        # Apply first control input to move each robot
        for bot in range(num_bots):
            if np.linalg.norm(initial_poses[bot] - final_poses[bot]) > 0.05:
                initial_poses[bot] = holonomoic_drive_model(initial_poses[bot], U_opt[bot, 0])
            x_ref[bot] = np.linspace(initial_poses[bot][0], final_poses[bot][0], N)
            y_ref[bot] = np.linspace(initial_poses[bot][1], final_poses[bot][1], N)
            ref_traj[bot] = np.stack([x_ref[bot], y_ref[bot]], axis=1) 

        # Replan for next step
        # print("initial pose  ", initial_poses)
        result = minimize(objective, U_opt.flatten(), method='SLSQP', bounds=bounds)
        U_opt = result.x.reshape(num_bots, N, 2)

    # print("bot 1 control : ", U_opt[0][0], "bot 2 control : ", U_opt[1][0])
    
if animate:
    x_all = [initial_poses[bot].copy() for bot in range(num_bots)]
    trajectory = [[] for _ in range(num_bots)]

    for bot in range(num_bots):
        x = x_all[bot].copy()
        trajectory[bot].append(x.copy())

        for t in range(N):
            x = holonomoic_drive_model(x, U_opt[bot][t])
            trajectory[bot].append(x.copy())

    trajectory = [np.array(traj) for traj in trajectory]

    # Plot
    plt.clf()
    colors = ['b', 'r', 'm', 'c', 'orange', 'purple']
    for bot in range(num_bots):
        traj = trajectory[bot]
        plt.plot(traj[:, 0], traj[:, 1], color=colors[bot % len(colors)], label=f'Bot {bot} MPC Path')
        plt.plot(ref_traj[bot][:, 0], ref_traj[bot][:, 1], '--', color=colors[bot % len(colors)], alpha=0.5)
        plt.scatter(initial_poses[bot][0], initial_poses[bot][1], marker='o', c=colors[bot % len(colors)], s=100)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('MPC Trajectories for Multiple Holonomic Robots')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()  # Non-blocking plot
    
    
    
