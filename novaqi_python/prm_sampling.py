import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import csv
import time
import os

# Parameters
N_SAMPLE = 500  # number of sample_points
N_KNN = 5  # number of edges from one sampled point
MAX_EDGE_LEN = 5.0  # [m] Maximum edge length
show_animation = True

def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)

    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])
        if dist <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK

def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    """
    Road map generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    print("Generating road map...")
    start_time = time.time()

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
        if i % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {i}/{n_sample} samples, elapsed time: {elapsed_time:.2f} seconds")

        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    total_time = time.time() - start_time
    print(f"Completed road map generation in {total_time:.2f} seconds")

    return road_map

def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover
    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")

def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    max_x = max(ox) 
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random

    print("Sampling points...")
    start_time = time.time()

    while len(sample_x) <= N_SAMPLE:
        if len(sample_x) % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Sampled {len(sample_x)}/{N_SAMPLE} points, elapsed time: {elapsed_time:.2f} seconds")

        tx = (rng.uniform() * (max_x - min_x)) + min_x
        ty = (rng.uniform() * (max_y - min_y)) + min_y

        dist, _ = obstacle_kd_tree.query([tx, ty])

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    total_time = time.time() - start_time
    print(f"Completed sampling points in {total_time:.2f} seconds")

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y

def occupancy_grid_to_obstacles(occupancy_grid, resolution):
    """
    Converts an occupancy grid to lists of obstacle coordinates.

    :param occupancy_grid: 2D array representing the occupancy grid
    :param resolution: size of each cell in the grid (meters)
    :return: two lists containing x and y coordinates of obstacles
    """
    obstacle_x = []
    obstacle_y = []
    for y in range(occupancy_grid.shape[0]):
        for x in range(occupancy_grid.shape[1]):
            if occupancy_grid[y, x] == 100:  # assuming 100 represents an obstacle
                obstacle_x.append(x * resolution)
                obstacle_y.append(y * resolution)
    return obstacle_x, obstacle_y

def main(rng=None):
    print("Starting script...")
    script_start_time = time.time()

    matrix_path = "real_map4_Ogrid.csv"

    with open(matrix_path, 'r') as f:
        reader = csv.reader(f)
        loaded_matrix = [list(map(float, row)) for row in reader]

    occupancy_grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])  # Example occupancy grid

    # Load occupancy grid and convert to obstacle lists
    resolution = 1  # Adjust this based on your grid's resolution (meters per cell)
    ox, oy = occupancy_grid_to_obstacles(occupancy_grid, resolution)

    print(f"Loaded occupancy grid with resolution {resolution}m")

    # Start and goal positions
    sx = 0.0  # [m]
    sy = 0.0  # [m]
    gx = 9.0  # [m]
    gy = 9.0  # [m]
    robot_size = 1.0  # [m]

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")

    obstacle_kd_tree = KDTree(np.vstack((ox, oy)).T)
    sample_x, sample_y = sample_points(sx, sy, gx, gy, robot_size, ox, oy, obstacle_kd_tree, rng)
    
    print(f"KDTree constructed and sample points generated")

    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(sample_x, sample_y, robot_size, obstacle_kd_tree)
    print(f"Road map generated")
    #print(road_map)
    
    if show_animation:
        plot_road_map(road_map, sample_x, sample_y)
        plt.pause(0.001)

        # Save the plot
        current_directory = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(current_directory, "road_map_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved as {plot_path}")

        plt.show()
    for i in range(503):
        if len(road_map[i])<4:
            print("SCENE HOGAYA")
            print(len(road_map[i]))
    total_script_time = time.time() - script_start_time
    print(f"Script completed in {total_script_time:.2f} seconds")

    

if __name__ == '__main__':
    main()
