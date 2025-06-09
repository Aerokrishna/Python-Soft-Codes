#!/usr/bin/python3
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import csv
import time
import os
from collections import defaultdict

# Parameters
N_SAMPLE = 1000  # number of sample_points
N_KNN = 5  # number of edges from one sampled point
MAX_EDGE_LEN = 5.0  # [m] Maximum edge length
show_animation = False
np.set_printoptions(threshold=np.inf)

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
def add_edge(graph, u, v):
    graph[u].append(v)
    graph[v].append(u)

def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    road_map = defaultdict(list)
    
    # road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    print("Generating road map...")
    start_time = time.time()

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
        if i % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {i}/{n_sample} samples, elapsed time: {elapsed_time:.2f} seconds")

        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edges = []

        for ii, neighbors in enumerate(indexes):
            if ii:
                nx = sample_x[indexes[ii]]
                ny = sample_y[indexes[ii]]

                if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                    add_edge(graph=road_map, u=i, v=neighbors)

                if len(road_map[i]) >= N_KNN:
                    print("parent  ", i, "children   ", road_map[i])

                    break
        
        # road_map[i] = edges

    total_time = time.time() - start_time
    print(f"Completed road map generation in {total_time:.2f} seconds")

    return road_map


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover
    visited = set()

    for i in road_map:
        for j in road_map[i]:
            if (j, i) in visited:  # avoid plotting the same undirected edge twice
                continue
            plt.plot([sample_x[i], sample_x[j]],
                     [sample_y[i], sample_y[j]], "-k")
            visited.add((i, j))

def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    max_x = max(ox) 
    max_y = max(oy) 
    min_x = min(ox) 
    min_y = min(oy) 

    # max_x = 40.0
    # max_y = 40.0 
    # min_x = 0.0 
    # min_y = 0.0 

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
            sample_x.append(round(tx,2))
            sample_y.append(round(ty,2))

    total_time = time.time() - start_time
    print(f"Completed sampling points in {total_time:.2f} seconds")

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
    for y in range(occupancy_grid.shape[1]):
        for x in range(occupancy_grid.shape[0]):
            if occupancy_grid[y, x] == 100:  # assuming 100 represents an obstacle
                obstacle_x.append(y * resolution)
                obstacle_y.append(x * resolution)
    return obstacle_x, obstacle_y

def plot_surface(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the bars
    ax.bar3d(x, y, np.zeros_like(z), 0.5, 0.5, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    plt.show()

def main(rng=None):
    print("Starting script...")
    script_start_time = time.time()

    matrix_path = "real_map4_Ogrid.csv"

    with open(matrix_path, 'r') as f:
        reader = csv.reader(f)
        loaded_matrix = [list(map(float, row)) for row in reader]

    occupancy_grid = np.array(loaded_matrix)
    # occupancy_grid = np.zeros((40,40))

    print(occupancy_grid)

    resolution = 1.0  # Adjust this based on your grid's resolution (meters per cell)
    ox, oy = occupancy_grid_to_obstacles(occupancy_grid, resolution)

    print(f"Loaded occupancy grid with resolution {resolution}m")

    # Start and goal positions
    sx = 0.0  # [m]
    sy = 0.0  # [m]
    gx = 20.0  # [m]
    gy = 20.0  # [m]
    
    robot_size = 1.0  # [m]

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^r")
        plt.grid(True)
        plt.axis("equal")

    obstacle_kd_tree = KDTree(np.vstack((ox, oy)).T)
    sample_x, sample_y = sample_points(sx, sy, gx, gy, robot_size, ox, oy, obstacle_kd_tree, rng)
    # print(sample_x)
    # print(sample_y)
    
    print(f"KDTree constructed and sample points generated")

    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(sample_x, sample_y, robot_size, obstacle_kd_tree)
    print(f"Road map generated")
    
    if show_animation:
        plot_road_map(road_map, sample_x, sample_y)
        plt.pause(0.001)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(current_directory, "road_map_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved as {plot_path}")

        plt.show()

    alpha = 0.3
    gamma = 0.95
    epsilon = 0.1
    num_episodes = 1200
    num_states = len(sample_x)
    # num_actions = N_KNN

    sample_set = list(range(num_states))

    Q_table = defaultdict(list)

    for i in road_map:
        degree = len(road_map[i])
        Q_table[i] = [0] * degree

    goal_state = len(sample_x) - 1
   
    for episode in range(num_episodes):
        print("episode : ", episode)
        current_state = np.random.choice(sample_set)
        cnt = 0
        
        while True:
            if np.random.rand() < epsilon:
                action_id = np.random.choice(range(len(road_map[current_state])))
                action = road_map[current_state][action_id]
            else:
                # print(current_state, Q_table[current_state])
                action_id = np.argmax(Q_table[current_state])
                # if action_id >= len(road_map[current_state]):
                #     break
                action = road_map[current_state][action_id]

            next_state = action

            if next_state == goal_state:
                reward = 100
            else:
                reward = -5
            
            Q_table[current_state][action_id] = Q_table[current_state][action_id] + alpha * (reward + gamma * max(Q_table[next_state]) - Q_table[current_state][action_id])
            current_state = next_state

            cnt += 1

            if reward == 100 or cnt == 100:
                break

    # Convert Q_table to z values for plotting
    gridMap_optimalQ_values = [max(Q_table[i]) for i in range(num_states)]

    # with open(f'/home/krishnapranav/novaqi_python/prm_qtable.csv', 'w', newline='') as f:
    #             writer = csv.writer(f)
    #             for row in zip(sample_x, sample_y, gridMap_optimalQ_values):
    #                 writer.writerow(row)
    
    #print(gridMap_optimalQ_values)
    plot_surface(sample_x, sample_y, gridMap_optimalQ_values)
    
    total_script_time = time.time() - script_start_time
    print(f"Script completed in {total_script_time:.2f} seconds")
    print(sample_x[len(sample_x) - 1])
if __name__ == '__main__':
    main()
