import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

# occupancyGrid = np.array([[100,100,100,100,100],
#                  [0,0,0,0,100],
#                  [0,0,0,0,0],
#                  [0,0,0,100,0],
#                  [100,100,0,0,0],
#                  [0,0,0,0,0],
#                  [100,100,100,0,0],
#                  [0,0,0,0,0],
#                  [100,0,0,0,0],
#                  [100,100,100,100,100]])

world_length = 4 # in meters
world_breadth = 4 # in meters

width = 80 # of image in pixels
height = 80 # width/height in pixels should be same asd width/height in meters 160 by 120 8 by 6

map_img = cv2.imread("sample_map.png", cv2.IMREAD_COLOR)

resized_img = cv2.resize(map_img, (width, height))

cell_size = 2 # that is one cell contains 2 rows and columns of pixels

# Ensure occupancyGrid matches resized image dimensions
occupancyGrid = np.zeros((height // cell_size, width // cell_size), dtype=np.int8)

# Define color thresholds (modify as needed)
threshold = 100

white_count = 0
black_count = 0

map_resolution = width/(cell_size * world_length) # cells per meter

for rows in range(0, height, cell_size):
    for columns in range(0, width, cell_size):

        for i in range(cell_size):
            for j in range(cell_size):
                pixel = resized_img[i + rows][j + columns]
                # Check all colors once per 4x4 block
               
                if pixel[0] < threshold and pixel[1] < threshold and pixel[2] < threshold:
                    white_count += 1
                else : 
                    black_count += 1
                
        dominant_color = max(white_count, black_count)
    
        cell_rows = rows // cell_size
        cell_columns = columns // cell_size

        if dominant_color == white_count:
            occupancyGrid[cell_rows][cell_columns] = 100  # Occupied
            white_count = 0
            black_count = 0
        else : 
            occupancyGrid[cell_rows][cell_columns] = 0  # Free space
            white_count = 0
            black_count = 0

np.set_printoptions(threshold=np.inf)

# print(occupancyGrid)

grid_shape = occupancyGrid.shape
num_states = grid_shape[0] * grid_shape[1]
num_actions = 4
alpha = 0.3 
gamma = 0.95 
epsilon = 0.1
num_episodes = 50000

sample_set = list(range(num_states))
Q_table = np.zeros((num_states,num_actions), dtype=np.float64)

gridMap_optimal_actions = np.zeros((grid_shape[0], grid_shape[1]), dtype=np.float64) 
gridMap_optimalQ_values = np.zeros((grid_shape[0], grid_shape[1]), dtype=np.float64) 

plot_arrayY = np.tile(np.arange(grid_shape[1]), (grid_shape[0], 1))
plot_arrayX = np.tile(np.arange(grid_shape[0]).reshape(grid_shape[0], 1), (1, grid_shape[1]))

goal_pose = (3.0,3.0)
goal_cell = (goal_pose[0] * map_resolution, goal_pose[1] * map_resolution)
goal_state = round(goal_cell[0] * grid_shape[1] + goal_cell[1])

print(goal_state)
# goal_state = 220
print(goal_cell)
print(grid_shape)

def check_and_update_reward(state_i, state_j, state_reward):
        if occupancyGrid[state_i, state_j] == 100:
            state_reward -= 50
        return state_reward

def simulate_action():
    global action
    (i,j) = np.unravel_index(current_state, occupancyGrid.shape)

    if action == 0:
        i = max(i - 1, 0) # manages boundary cases very useful !!
    if action == 1:
        i = min(i + 1, grid_shape[0]-1)
    if action == 2:
        j = max(j - 1, 0)
    if action == 3:
        j = min(j + 1, grid_shape[1]-1)
    
    next_state = i * grid_shape[1] + j # convert it into state number

    if occupancyGrid[i][j] == 100:
        reward = -500 # obstacle in next_state
    elif next_state == goal_state:
        reward = 100 # goal is reached
    else:
        reward = - 5

    # for giving low rewards if end up in a cell next to an obstacle
    if reward == -5:
        # Compute initial (i, j) from next_state
        i, j = np.unravel_index(next_state, occupancyGrid.shape)

        # Precompute potential movements
        movements = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1)    # right
        ]
        # Process primary movements
        for di, dj in movements:
            ni = np.clip(i + di, 0, grid_shape[0] - 1)
            nj = np.clip(j + dj, 0, grid_shape[1] - 1)
            reward = check_and_update_reward(ni, nj, reward)

            # Process secondary movements
            for ddi, ddj in movements:
                nni = np.clip(ni + ddi, 0, grid_shape[0] - 1)
                nnj = np.clip(nj + ddj, 0, grid_shape[1] - 1)
                reward = check_and_update_reward(nni, nnj, reward)
    
        # for r in range(num_actions):
        #     # print("SPECIAL CASE !!!")
        #     (i,j) = np.unravel_index(next_state, occupancyGrid.shape)
        #     if r == 0:

        #         i = max(i - 1, 0) # manages boundary cases very useful !!
        #         next_to_nextstate = i * grid_shape[1] + j
        #         for r in range(num_actions):
        #             # print("SPECIAL CASE !!!")
        #             (i,j) = np.unravel_index(next_to_nextstate, occupancyGrid.shape)
        #             if r == 0:
        #                 i = max(i - 1, 0) # manages boundary cases very useful !!
        #             if r == 1:
        #                 i = min(i + 1, grid_shape[0]-1)
        #             if r == 2:
        #                 j = max(j - 1, 0)
        #             if r == 3:
        #                 j = min(j + 1, grid_shape[1]-1)

        #             if occupancyGrid[i][j] == 100:
        #                 # print(occupancyGrid[i][j])
                        
        #                 reward = reward - 50

        #     if r == 1:
        #         i = min(i + 1, grid_shape[0]-1)
        #         next_to_nextstate = i * grid_shape[1] + j
        #         for r in range(num_actions):
        #             # print("SPECIAL CASE !!!")
        #             (i,j) = np.unravel_index(next_to_nextstate, occupancyGrid.shape)
        #             if r == 0:
        #                 i = max(i - 1, 0) # manages boundary cases very useful !!
        #             if r == 1:
        #                 i = min(i + 1, grid_shape[0]-1)
        #             if r == 2:
        #                 j = max(j - 1, 0)
        #             if r == 3:
        #                 j = min(j + 1, grid_shape[1]-1)

        #             if occupancyGrid[i][j] == 100:
        #                 # print(occupancyGrid[i][j])
                        
        #                 reward = reward - 50
        #     if r == 2:
        #         j = max(j - 1, 0)
        #         next_to_nextstate = i * grid_shape[1] + j
        #         for r in range(num_actions):
        #             # print("SPECIAL CASE !!!")
        #             (i,j) = np.unravel_index(next_to_nextstate, occupancyGrid.shape)
        #             if r == 0:
        #                 i = max(i - 1, 0) # manages boundary cases very useful !!
        #             if r == 1:
        #                 i = min(i + 1, grid_shape[0]-1)
        #             if r == 2:
        #                 j = max(j - 1, 0)
        #             if r == 3:
        #                 j = min(j + 1, grid_shape[1]-1)

        #             if occupancyGrid[i][j] == 100:
        #                 # print(occupancyGrid[i][j])
                        
        #                 reward = reward - 50
        #     if r == 3:
        #         j = min(j + 1, grid_shape[1]-1)
        #         next_to_nextstate = i * grid_shape[1] + j
        #         for r in range(num_actions):
        #             # print("SPECIAL CASE !!!")
        #             (i,j) = np.unravel_index(next_to_nextstate, occupancyGrid.shape)
        #             if r == 0:
        #                 i = max(i - 1, 0) # manages boundary cases very useful !!
        #             if r == 1:
        #                 i = min(i + 1, grid_shape[0]-1)
        #             if r == 2:
        #                 j = max(j - 1, 0)
        #             if r == 3:
        #                 j = min(j + 1, grid_shape[1]-1)

        #             if occupancyGrid[i][j] == 100:
        #                 # print(occupancyGrid[i][j])
                        
        #                 reward = reward - 50

        #     (i,j) = np.unravel_index(next_state, occupancyGrid.shape)
        #     if occupancyGrid[i][j] == 100:
        #         # print(occupancyGrid[i][j])
                
        #         reward = reward - 100
        #         # print(reward)
        #     else:
        #         reward = reward
                

    # print(reward)
    return next_state,reward

def plot_grid(X,Y,Z):
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="Spectral")

    # plt.pause(0.01)

for episode in range(num_episodes):
    if episode == 10000 or episode == 50000 or episode == 75000 or episode == 90000:
        print("episode  ",episode)
    print("episode  ",episode)
    current_state = np.random.choice(sample_set)

    (i,j) = np.unravel_index(current_state, occupancyGrid.shape)

    if occupancyGrid[i][j] == 100:
        Q_table[current_state] = [-500, -500, -500, -500]
        element_to_remove = current_state
        sample_set = [x for x in sample_set if x != element_to_remove] # creates a new sample set excluding the element to remove
        continue

    while True:
        # explore
        if np.random.rand() < epsilon:
            action =  np.random.randint(0,num_actions)
        # exploit
        else:
            action = np.argmax(Q_table[current_state])
 
        # action =  np.random.randint(0,num_actions)
        next_state, reward = simulate_action()
        
        Q_table[current_state][action] = Q_table[current_state][action] + alpha * (reward + gamma * max(Q_table[next_state]) - Q_table[current_state][action])

        current_state = next_state
     
        if reward == 100 or reward == -500:
            break
    
    rowMax = np.max(Q_table, axis=1) 
    maxIndices = np.argmax(Q_table, axis=1) 
    for k in range(num_states):
        (m,n) = np.unravel_index(k, occupancyGrid.shape)
        gridMap_optimal_actions[m][n] = maxIndices[k]
        gridMap_optimalQ_values[m][n] = rowMax[k]

plot_grid(X = plot_arrayX, Y = plot_arrayY, Z = gridMap_optimalQ_values)
plt.show()

np.set_printoptions(threshold=np.inf)

print(gridMap_optimalQ_values)
print(gridMap_optimal_actions)
print(occupancyGrid)

# Save the matrix to a CSV file
with open('qlearning_Ogrid.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(gridMap_optimalQ_values)
