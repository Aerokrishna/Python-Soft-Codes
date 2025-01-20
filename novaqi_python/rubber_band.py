import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Simulated Occupancy Grid and Lidar Data
def load_occupancy_grid():
    """Simulate the occupancy grid as a numpy array."""
    grid_size = (50, 50)  # Example grid size
    occupancy_grid = np.zeros(grid_size)  # Empty grid (0 = free, 1 = obstacle)
    # Add obstacles
    occupancy_grid[20:30, 20:30] = 1  # Example square obstacle
    return occupancy_grid

def simulate_lidar(position, grid, max_range=500, num_rays=360):
    """Simulate LiDAR range readings from a given position."""
    angles = np.linspace(0, 2 * np.pi, num_rays)
    ranges = np.zeros(num_rays)
    for i, angle in enumerate(angles):
        for r in np.linspace(0, max_range, 1000):  # Incremental range
            x = int(position[0] + r * np.cos(angle))
            y = int(position[1] + r * np.sin(angle))
            if x < 0 or y < 0 or x >= grid.shape[0] or y >= grid.shape[1] or grid[x, y] == 1:
                ranges[i] = r
                break
    return ranges, angles

def calculate_r_max(ranges, angles):
    """Calculate R_max (maximum free radius) for each direction."""
    return ranges  # In this simplified case, R_max equals LiDAR ranges.

# Load data
occupancy_grid = load_occupancy_grid()
P = np.array([24, 24])  # Robot's position

# Simulate LiDAR data
ranges, angles = simulate_lidar(P, occupancy_grid)

# Compute R_max
R_max = calculate_r_max(ranges, angles)

# Convert polar to Cartesian coordinates
lidar_points = np.array([P[0] + ranges * np.cos(angles), P[1] + ranges * np.sin(angles)]).T
rmax_points = np.array([P[0] + R_max * np.cos(angles), P[1] + R_max * np.sin(angles)]).T

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Visualization 1: Lidar Ranges and R_max
ax[0].imshow(occupancy_grid.T, origin='lower', cmap='gray', extent=[0, occupancy_grid.shape[0], 0, occupancy_grid.shape[1]])
ax[0].plot(P[0], P[1], 'ro', label='Robot Position')
ax[0].plot(lidar_points[:, 0], lidar_points[:, 1], 'r-', label='LiDAR Range')
ax[0].plot(rmax_points[:, 0], rmax_points[:, 1], 'm-', label='R_max')
ax[0].set_title('LiDAR Data and R_max')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].axis('equal')
ax[0].legend()

# Visualization 2: Heatmap and Optimal Points
# Create a mock gridMap_optimal_values
gridMap_optimal_values = np.random.rand(*occupancy_grid.shape) * (1 - occupancy_grid)

# Determine IPC limits (Intermediate Potential Candidate limits)

R_max_array = np.array(R_max)  # Ensure R_max is an array
IPC_limits = np.round(P[:, None] + np.array([R_max_array * np.cos(angles), 
                                             R_max_array * np.sin(angles)])).T

# Find the optimal point
best_val = -np.inf
opt_target = None
for ipc in IPC_limits:
    x, y = ipc
    x=int(x)
    y=int(y)
    if 0 <= x < gridMap_optimal_values.shape[0] and 0 <= y < gridMap_optimal_values.shape[1]:
        if gridMap_optimal_values[x, y] > best_val:
            best_val = gridMap_optimal_values[x, y]
            opt_target = ipc

# Heatmap visualization
ax[1].imshow(gridMap_optimal_values.T, origin='lower', cmap='hot', extent=[0, occupancy_grid.shape[0], 0, occupancy_grid.shape[1]])
ax[1].plot(P[0], P[1], 'ro', label='Robot Position')
ax[1].plot(rmax_points[:, 0], rmax_points[:, 1], 'm-', label='R_max')
ax[1].scatter(IPC_limits[:, 0], IPC_limits[:, 1], c='b', s=10, label='IPC Limits')
if opt_target is not None:
    ax[1].plot(opt_target[0], opt_target[1], 'co', label='Optimal Target')
ax[1].set_title('Occupancy Grid and Optimal Waypoints')
ax[1].set_xlabel('X')
ax[1].set_ylabel('Y')
ax[1].axis('equal')
ax[1].legend()

plt.tight_layout()
plt.show()
