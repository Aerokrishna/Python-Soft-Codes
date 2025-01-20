import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt

def costmap(occupancyGrid):
    actions = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
    shape = occupancyGrid.shape

    occupied_cells = np.argwhere(occupancyGrid == 1)
    # for m in range(shape[0]):
    #     for n in range(shape[1]):
            
            #if occupancyGrid[m][n] == 1:

                # for i in range(len(actions)):
                #     cell_rows = m
                #     cell_columns = n

                #     cell_rows += actions[i][0]
                #     cell_columns += actions[i][1]
                #     if (cell_rows < occupancyGrid.shape[0]-1 and cell_rows > 0) and (cell_columns < occupancyGrid.shape[1]-1 and cell_columns > 0):

                #         if occupancyGrid[cell_rows][cell_columns] == 1:
                #             continue
                #         else:
                #             occupancyGrid[cell_rows][cell_columns] = 0.3

    # Iterate over all occupied cells
    # print(occupied_cells)
    for m, n in occupied_cells:
        # print(m,n)
        # Calculate the neighboring cells
        neighbors = actions + [m, n] # adding two matrices, basically getting a list of all the neighbouring 

        # conditionally check the neighbours
        valid_neighbors = neighbors[(neighbors[:, 0] >= 0) & (neighbors[:, 0] < shape[0]) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < shape[1])]

        occupancyGrid[valid_neighbors[:, 0], valid_neighbors[:, 1]] = np.where(
            occupancyGrid[valid_neighbors[:, 0], valid_neighbors[:, 1]] == 1, 
            1, 
            0.5
        )

    return occupancyGrid

def costmap_2(occupancyGrid):
    actions = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
    shape = occupancyGrid.shape

    occupied_cells = np.argwhere(occupancyGrid == 0.5)
 
    for m, n in occupied_cells:

        neighbors = actions + [m, n] # adding two matrices, basically getting a list of all the neighbouring 

        # conditionally check the neighbours
        valid_neighbors = neighbors[(neighbors[:, 0] >= 0) & (neighbors[:, 0] < shape[0]) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < shape[1])]

        occupancyGrid[valid_neighbors[:, 0], valid_neighbors[:, 1]] = np.where(
            occupancyGrid[valid_neighbors[:, 0], valid_neighbors[:, 1]] == 0 , 
            0.3, 
            occupancyGrid[valid_neighbors[:, 0], valid_neighbors[:, 1]]
        ) # if condition is true replaces it with 1 else 0.3

    return occupancyGrid

def occupancy_grid(resized_img, cell_size, threshold):

    height, width, _ = resized_img.shape
    occupancyGrid = np.zeros((height // cell_size, width // cell_size), dtype=np.float32)
    
    # Reshape and aggregate the image
    reshaped_img = resized_img.reshape(height // cell_size, cell_size, width // cell_size, cell_size, 3)
    
    # Calculate mean color in each cell block
    mean_colors = reshaped_img.mean(axis=(1, 3))  # Averaging over the cell_size dimension
    
    # Determine if the block is "white" or "black" based on the threshold
    white_mask = np.all(mean_colors < threshold, axis=-1)  # Check if all color channels are below the threshold
    occupancyGrid[white_mask] = 1  # Occupied by white color
    return occupancyGrid

np.set_printoptions(threshold=np.inf)

start_coord = [20,20]
goal_coord = [78,50]

# start_coord = [0,0]
# goal_coord = [2,1]

world_length = 4
# image_path = "/home/algs/NavAlgorithms/sample_map_1.pgm"
map_img = cv2.imread('sample_map_1.png', cv2.IMREAD_COLOR)
print(map_img)
width = (np.round(map_img.shape[1] / 10) * 10).astype(int)# of image in pixels
height = (np.round(map_img.shape[0] / 10) * 10).astype(int) # width/height in pixels should be same asd width/height in meters 160 by 120 8 by 6

map_resolution = 20 # cells per meter, currently 20 pixels per meter
resized_img = cv2.resize(map_img, (width, height))
cell_size = int(width/(world_length * map_resolution))

occupancyGrid = occupancy_grid(resized_img, cell_size, 150)
# print(occupancyGrid)

# occupancyGrid = np.array([[0,0,0,0,0],
#                           [0,0,0,0,0],
#                           [0,0,1,0,0],
#                           [0,0,0,0,0],
#                           [0,0,0,0,0],], dtype='float32')
# shape = occupancyGrid.shape
occupancyGrid = costmap(occupancyGrid)
occupancyGrid = costmap_2(occupancyGrid)
print(occupancyGrid)

ogrid_path = 'map_Ogrid.csv'
with open(ogrid_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(occupancyGrid)

grid_size = occupancyGrid.shape
print(grid_size)

start_node = start_coord[0] * grid_size[1] + start_coord[1]
goal_node = goal_coord[0] * grid_size[1] + goal_coord[1]

open_nodes = {start_node : (np.inf,0)}
closed_nodes = {}
parent_nodes = {}

ideal_nodes = []

def is_valid(cell_x,cell_y,occupancyGrid, closed_nodes):
    next_node = cell_x * occupancyGrid.shape[1] + cell_y

    if next_node in closed_nodes:
        return False
    
    elif cell_x < 0 or cell_x >= occupancyGrid.shape[0] or cell_y < 0 or cell_y >= occupancyGrid.shape[1]:
        return False
    
    elif occupancyGrid[cell_x][cell_y] >= 0.3:
        return False
    
    else:
        return True

def get_distance(x1,y1,x2,y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_cost(node,occupancyGrid,goal,start):
    (i,j) = np.unravel_index(node, occupancyGrid.shape)
    (gi,gj) = np.unravel_index(goal, occupancyGrid.shape)
    (si,sj) = np.unravel_index(start, occupancyGrid.shape)
    
    h_cost = get_distance(i,j,gi,gj)
    g_cost = get_distance(i,j,si,sj)

    f_cost = h_cost + g_cost

    return f_cost,g_cost

def retrace_path(goal_node, start_node):
    current_node = goal_node
    while current_node != start_node:
        parent_node = parent_nodes[current_node]
        current_node = parent_node
        ideal_nodes.insert(0,current_node)
        # print(current_node)
    
    print('TRACED PATH!')
    
def a_star(start_node, goal_node, occupancyGrid):
    global closed_nodes
    global open_nodes
    global ideal_nodes
    global parent_nodes

    current_node = start_node
    actions = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]

    while True:
        min_fcost = np.inf
        for i in range(len(actions)):
            current_node_x, current_node_y = np.unravel_index(current_node, occupancyGrid.shape)
            
            current_node_x += actions[i][0]
            current_node_y += actions[i][1]
            
        
            if is_valid(current_node_x,current_node_y,occupancyGrid, closed_nodes):

                next_node = current_node_x * occupancyGrid.shape[1] + current_node_y

                # print(current_node_x,current_node_y)
                f_cost,g_cost = get_cost(next_node,occupancyGrid,goal_node,start_node)
                # print(f_cost)

                if next_node not in open_nodes or open_nodes[next_node][0] > f_cost:
                    open_nodes[next_node] = (f_cost,g_cost)
                    parent_nodes[next_node] = current_node

            else:
                continue
            
            if f_cost < min_fcost:
                min_fcost = f_cost
                min_node = current_node_x * occupancyGrid.shape[1] + current_node_y


                # print(min_node)
            
            # elif f_cost == min_fcost:
            #     min_fcost = 
        
        # remove the node from open nodes and keep it in closed
        closed_nodes[current_node] = open_nodes.pop(current_node)

        current_node = min_node

        if current_node == goal_node:
            retrace_path(goal_node,start_node)
            print("GOAL REACHED : ", current_node)
            break

        # print("nodes traversed : ", current_node)

a_star(start_node,goal_node,occupancyGrid)
plt.imshow(occupancyGrid, cmap='gray_r', interpolation='nearest')

closed_nodes_coord = []
open_nodes_coord = []
ideal_nodes_coord = []
ideal_nodes_coord = []
ideal_nodes_x = []
ideal_nodes_y = []

for i in closed_nodes:
    (x,y) = np.unravel_index(i, occupancyGrid.shape)
    closed_nodes_coord.append((x,y))

for j in open_nodes:
    (x,y) = np.unravel_index(j, occupancyGrid.shape)
    open_nodes_coord.append((x,y))

for k in ideal_nodes:
    (x,y) = np.unravel_index(k, occupancyGrid.shape)
    ideal_nodes_coord.append((x,y))
    ideal_nodes_x.append(x)
    ideal_nodes_y.append(y)


for coord in closed_nodes_coord:
    plt.scatter(coord[1], coord[0], color='red', s=100, marker='s')  # `s` controls the size of the points

for coord in open_nodes_coord:
    plt.scatter(coord[1], coord[0], color='green', s=100, marker='s')  # `s` controls the size of the points

for coord in ideal_nodes_coord:
    plt.scatter(coord[1], coord[0], color='yellow', s=100, marker='s')  # `s` controls the size of the points

plt.scatter(start_coord[1], start_coord[0], color='orange', s=100, marker='s')
plt.scatter(goal_coord[1], goal_coord[0], color='blue', s=100, marker='s')

print(parent_nodes)
plt.show()

# print(f_cost_nodes)
# print(open_nodes)
print(ideal_nodes_x)











