#!/usr/bin/env python3
import heapq
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class Astar_Node:
    def __init__(self, x, y, cost, time, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent
        self.time = time

    def __lt__(self, other):
        return self.cost < other.cost

def distance(node1, node2):
    return np.hypot(node1.x - node2.x, node1.y - node2.y)

def is_collision_free(node, occupancy_grid, path_matrix, time, error=1):
    x, y = int(node.x), int(node.y)
    if x < 0 or y < 0 or x >= occupancy_grid.shape[1] or y >= occupancy_grid.shape[0]:
        return False
    return occupancy_grid[y, x] == 100 and all(t not in path_matrix[y, x] for t in range(int(time)-int(error), int(time)+int(error)+1))

def path(node, path_matrix, buffer=1):
    p = [node]
    while node.parent is not None:
        node = node.parent
        for i in range(-buffer, buffer + 1):
            for j in range(-buffer, buffer + 1):
                if 0 <= node.y + i < path_matrix.shape[0] and 0 <= node.x + j < path_matrix.shape[1]:
                    if node.time not in path_matrix[node.y+i, node.x+j]:
                        path_matrix[node.y+i, node.x+j].append(node.time)
        p.append(node)
    return p[::-1]

def a_star(start, goal, occupancy_grid, bot_id=1, path_matrix=None):
    start_node = Astar_Node(start[0], start[1], 0, 0.0)
    goal_node = Astar_Node(goal[0], goal[1], -1, 0.0)

    open_set = []
    heapq.heappush(open_set, (0, start_node))
    closed_set = set()
    movements = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if (current.x, current.y) in closed_set:
            continue
        closed_set.add((current.x, current.y))

        if current.x == goal_node.x and current.y == goal_node.y:
            return path(current, path_matrix)

        for move in movements:
            new_x, new_y, new_t = current.x + move[0], current.y + move[1], current.time + 1
            new_node = Astar_Node(new_x, new_y, current.cost + distance(current, Astar_Node(new_x, new_y, 0, 0)), new_t, current)

            if not is_collision_free(new_node, occupancy_grid, path_matrix, new_t) or (new_x, new_y) in closed_set:
                continue

            heapq.heappush(open_set, (new_node.cost + distance(new_node, goal_node), new_node))

    return None

def create_ogrid():
    image_path = os.path.join('real_map4.png') 
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    occupancy_grid = np.zeros((height, width), dtype=np.uint8)
    
    reshaped_img = image.reshape(height , 1, width , 1, 3)
    mean_colors = reshaped_img.mean(axis=(1, 3))
    white_mask = np.all(mean_colors > 150, axis=-1)
    occupancy_grid[white_mask] = 100
    return occupancy_grid

def main():
    occupancy_grid = create_ogrid()
    H, W = occupancy_grid.shape
    print(occupancy_grid)
    pixel_data = np.empty((H, W), dtype=object)
    for i in range(H): 
        for j in range(W): 
            pixel_data[i, j] = []

    start = [(10, 10), (20, 30), (10, 30)]
    goal = [(40, 70), (70, 40), (60, 10)]

    path_result = [a_star(start[i], goal[i], occupancy_grid, i, pixel_data) for i in range(len(start))]

    for i, path in enumerate(path_result):
        print(f"\nBot {i+1} Path:")
        if path is not None:
            path_array = np.array([[node.x, node.y] for node in path])
            print(path_array)
        else:
            print("No path found.")
    # Plotting
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    plt.figure(figsize=(10, 10))
    plt.imshow(occupancy_grid, cmap='gray', origin='upper')

    for idx, path in enumerate(path_result):
        if path is None:
            print(f"Robot {idx+1}: Path not found.")
            continue
        x_vals = [node.x for node in path]
        y_vals = [node.y for node in path]
        plt.plot(x_vals, y_vals, color=colors[idx % len(colors)], label=f'Robot {idx+1}')
        plt.scatter([x_vals[0]], [y_vals[0]], c='black', marker='o')  # Start
        plt.scatter([x_vals[-1]], [y_vals[-1]], c='black', marker='x')  # Goal

    plt.legend()
    plt.title("Multi-Robot Spatiotemporal A* Paths")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()
    
    

if __name__ == '__main__':
    main()
