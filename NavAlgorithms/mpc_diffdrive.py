import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Simulation parameters
N = 10          # prediction horizon
dt = 0.1        # time step

# Cost function weights
Q = np.diag([10.0, 10.0, 1.0])  # penalize x, y, theta errors
R = np.diag([1.0, 1.0])         # penalize control effort [v, omega]

# Reference trajectory (straight line along x-axis)
x_ref = np.linspace(0, 1, N)
y_ref = np.zeros(N)
theta_ref = np.zeros(N)
ref_traj = np.stack([x_ref, y_ref, theta_ref], axis=1)
print(ref_traj)

# Initial state: [x, y, theta]
x0 = np.array([0.0, -0.5, 0.0])

# Differential drive motion model
def diff_drive_model(x, u):
    x_new = np.zeros_like(x)
    v, omega = u
    x_new[0] = x[0] + v * np.cos(x[2]) * dt
    x_new[1] = x[1] + v * np.sin(x[2]) * dt
    x_new[2] = x[2] + omega * dt
    return x_new

# Cost function for optimizer
def objective(U_flat):
    U = U_flat.reshape(N, 2) # flat array of control inputs
    x = x0.copy()
    cost = 0.0
    for t in range(N):
        x = diff_drive_model(x, U[t])
        x_err = x - ref_traj[t]
        u_err = U[t]

        # summation step using the formula 
        cost += x_err.T @ Q @ x_err + u_err.T @ R @ u_err
    return cost

# Initial guess for control inputs: [v, omega] for N steps, as 0
U0 = np.zeros((N, 2)).flatten()

# Solve the optimization problem
#It tries new guesses 
# of U_flat → feeds them into objective() → gets the cost → and uses optimization 
# logic (gradients, constraints, etc.) to improve the guess.

# U0 is a flat numpy array this will be the vector of decision variables the optimizer tries to change.
result = minimize(objective, U0, method='SLSQP')
U_opt = result.x.reshape(N, 2)

# Simulate with optimal control sequence
x = x0.copy()
trajectory = [x.copy()]
for t in range(N):
    x = diff_drive_model(x, U_opt[t])
    trajectory.append(x.copy())
trajectory = np.array(trajectory)
print(trajectory)
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
