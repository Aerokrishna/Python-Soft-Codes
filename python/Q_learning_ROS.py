import cv2
import numpy as np
import matplotlib.pyplot as plt

width = 80
height = 60 # width/height in pixels should be same asd width/height in meters

map_img = cv2.imread("my_map1.pgm", cv2.IMREAD_COLOR)
# grey_map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

resized_img = cv2.resize(map_img, (width, height))

cell_size = 2 # that is one cell contains 2 rows and columns of pixels

# Ensure occupancyGrid matches resized image dimensions
occupancyGrid = np.zeros((height // cell_size, width // cell_size), dtype=np.int8)

# Define color thresholds (modify as needed)
threshold = 100

map_resolution = 10 # cells per meter
action = -1

for rows in range(0, height, cell_size):
    for columns in range(0, width, cell_size):

        white_count = 0
        black_count = 0

        for i in range(cell_size):
            for j in range(cell_size):
                pixel = resized_img[i + rows][j + columns]
                # Check all colors once per 4x4 block
               
                if pixel[0] < threshold and pixel[1] < threshold and pixel[2] < threshold:
                    white_count += 1
                else : 
                    black_count += 1
                

        dominant_color = max(white_count, black_count)
        # print(no_count)
        cell_rows = rows // cell_size
        cell_columns = columns // cell_size

        if dominant_color == white_count:
            occupancyGrid[cell_rows][cell_columns] = 100  # Occupied
        else : 
            occupancyGrid[cell_rows][cell_columns] = 0  # Free space

np.set_printoptions(threshold=np.inf)
# print(occupancyGrid.shape)
# print(occupancyGrid)


goal = (2,3) # in meters
goal = (goal[0] * map_resolution, goal[1] * map_resolution)
goal_state = 45

grid_shape = occupancyGrid.shape # height * width...rows * columns
num_states = grid_shape[0] * grid_shape[1]
num_actions = 4
alpha = 0.3 # learning factor
gamma = 0.95 # discount factor
epsilon = 0.1 # exploration and exploitatio trade off
num_episodes = 5000

sample_set = list(range(num_states))

Q_table = np.zeros((num_states,num_actions), dtype=np.int8)

gridMap_optimal_actions = np.zeros((grid_shape[0], grid_shape[1]), dtype=np.int8) # stores optimal direction to move in the grid
gridMap_optimalQ_values = np.zeros((grid_shape[0], grid_shape[1]), dtype=np.int8) # stores the optimal q values in the grid

# plot_arrayX = np.zeros((grid_shape[0],grid_shape[1]),dtype=np.int8)
# plot_arrayY = np.zeros((grid_shape[0],grid_shape[1]),dtype=np.int8)

# for i,j in range(len(gridMap_optimalQ_values[0]),len(gridMap_optimalQ_values[1])):

#     plot_arrayX[j] = [j] * grid_shape[0]
#     plot_arrayY[i] = [i] * grid_shape[1]

def simulate_action():
    global action
    (i,j) = np.unravel_index(current_state, occupancyGrid.shape)
    if action == 0:
        i = max(i - 1, 1) # manages boundary cases very useful !!
    if action == 1:
        i = min(i + 1, grid_shape[0])
    if action == 2:
        j = max(j - 1, 1)
    if action == 3:
        j = min(j + 1, grid_shape[1])
    
    next_state = i * grid_shape[1] + j # convert it into state number
    
    # here i and j will correspond to the next state
    if occupancyGrid[i][j] == 100:
        reward = -500 # obstacle in next_state
    elif next_state == goal_state:
        reward = 100 # goal is reached
    else:
        reward = -10
    
    # # for giving low rewards if end up in a cell next to an obstacle
    # for action in range(num_actions):
    #     if action == 0:
    #         i = max(i - 1, 1) # manages boundary cases very useful !!
    #     if action == 1:
    #         i = min(i + 1, grid_shape[0]-1)
    #     if action == 2:
    #         j = max(j - 1, 1)
    #     if action == 3:
    #         j = min(j + 1, grid_shape[1]-1)
      
    #     if occupancyGrid[i][j] == 100:
    #         reward = reward - 50
            
    return next_state,reward

def plot_grid(X,Y,Z):
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="Spectral")

    plt.pause(0.01)

 # main loop
for episode in range(num_episodes):
    print("episode  ",episode)
    
    current_state = np.random.choice(sample_set) # does not include the last element
    # print("current state  ", current_state )
    (i,j) = np.unravel_index(current_state, occupancyGrid.shape)

    if occupancyGrid[i][j] == 100: # if occupied
        # remove from sample set    
        Q_table[current_state] = [-500, -500, -500, -500]
        element_to_remove = current_state
        sample_set = [x for x in sample_set if x != element_to_remove] # creates a new sample set excluding the element to remove
        continue # continues to the next iteration without executing while true

    while True:
        # # explore
        # if np.random.rand() < epsilon:
        #     print("EXPLORE!")
        #     action =  np.random.randint(0,num_actions)
            
        # else:
        #     max_Q_value = max(Q_table[current_state])
        #     # find its corresdpoding action
        #     action = np.where(Q_table[current_state] == max_Q_value)[0]
        #     print("No EXPLORE!")
        action =  np.random.randint(0,num_actions)

        # simulate the chosen action
        next_state, reward = simulate_action()

        # update Q value
        Q_table[current_state][action] = Q_table[current_state][action] + alpha * (reward + gamma * max(Q_table[current_state]) - Q_table[current_state][action])

        current_state = next_state

        if reward == 100 or reward == -500:
            break

    
    rowMax = np.max(Q_table, axis=1) # gets an array of maximum values from each column of the matrix
    maxIndices = np.argmax(Q_table, axis=1) # gets the index of that max value
    for k in range(num_states):
        (m,n) = np.unravel_index(k, occupancyGrid.shape)
        gridMap_optimal_actions[m][n] = maxIndices[k]
        gridMap_optimalQ_values[m][n] = rowMax[k]
    
    # plot_grid(X = plot_arrayX, Y = plot_arrayY, Z = gridMap_optimalQ_values)
    # plt.show
print(gridMap_optimalQ_values)
print(gridMap_optimal_actions)
print(occupancyGrid)
 

cv2.imshow("image", map_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


       



        
        


 

        




