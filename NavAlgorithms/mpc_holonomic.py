import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Simulation parameters
N = 10          # prediction horizon
dt = 0.1        # time step

# Cost function weights
Q = np.diag([10.0, 10.0, 1.0])  # penalize x, y, theta errors
R = np.diag([1.0, 1.0, 1.0])         # penalize control effort [v, omega]
W_dir = 1.0

# Reference trajectory (straight line along x-axis)
x_ref = np.linspace(0, 1, N)
y_ref = np.zeros(N)
theta_ref = np.zeros(N)
ref_traj = np.stack([x_ref, y_ref, theta_ref], axis=1)
# print(ref_traj)

# Initial state: [x, y, theta]
x0 = np.array([0.0, -0.5, 0.0])

# Differential drive motion model
def holonomoic_drive_model(x, u):
    x_new = np.zeros_like(x)
    vx, vy, omega = u
    x_new[0] = x[0] + vx * dt
    x_new[1] = x[1] + vy * dt
    x_new[2] = x[2] + omega * dt
    return x_new

# Cost function for optimizer
def objective(U_flat):
    U = U_flat.reshape(N, 3)  # [vx, vy, omega]
    x = x0.copy()
    cost = 0.0

    prev_theta = np.arctan2(U[0][1], U[0][0])  # Angle of initial velocity vector

    for t in range(N):
        # 1. Simulate forward
        x = holonomoic_drive_model(x, U[t])
        x_err = x - ref_traj[t]

        # 2. Control effort, penalizes large controls
        u_err = U[t]

        # 3. Penalize direction change
        vx, vy = U[t][0], U[t][1]
        theta = np.arctan2(vy, vx)
        dir_error = (theta - prev_theta)**2
        prev_theta = theta
        # print(x_err,u_err,dir_error)
        # 4. Total cost
        cost += x_err.T @ Q @ x_err + u_err.T @ R @ u_err + W_dir * dir_error

    return cost


# Initial guess for control inputs: [v, omega] for N steps, as 0
# U0 = np.zeros((N, 3)).flatten() # flat array of control inputs
U0 = np.random.rand(N, 3)
U0[:,2] = np.zeros((N))
# objective(U0)

# Solve the optimization problem
#It tries new guesses 
# of U_flat → feeds them into objective() → gets the cost → and uses optimization 
# logic (gradients, constraints, etc.) to improve the guess.

# U0 is a flat numpy array this will be the vector of decision variables the optimizer tries to change.
result = minimize(objective, U0.flatten(), method='SLSQP')
U_opt = result.x.reshape(N, 3)

# # Simulate with optimal control sequence
x = x0.copy()
trajectory = [x.copy()]
for t in range(N):
    x = holonomoic_drive_model(x,U_opt[t])
    trajectory.append(x.copy())
trajectory = np.array(trajectory)

print(U_opt)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(x_ref, y_ref, 'g--', label='Reference Path')
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='MPC Path')
plt.scatter(x0[0], x0[1], c='r', label='Start')
plt.xlabel('x')
plt.ylabel('y')
plt.title('MPC Trajectory Following (Quadratic Cost)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
