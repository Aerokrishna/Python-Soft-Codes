
        for r in range(num_actions):
            # print("SPECIAL CASE !!!")
            (i,j) = np.unravel_index(next_state, occupancyGrid.shape)
            if r == 0:

                i = max(i - 1, 0) # manages boundary cases very useful !!
                next_to_nextstate = i * grid_shape[1] + j
                for r in range(num_actions):
                    # print("SPECIAL CASE !!!")
                    (i,j) = np.unravel_index(next_to_nextstate, occupancyGrid.shape)
                    if r == 0:
                        i = max(i - 1, 0) # manages boundary cases very useful !!
                    if r == 1:
                        i = min(i + 1, grid_shape[0]-1)
                    if r == 2:
                        j = max(j - 1, 0)
                    if r == 3:
                        j = min(j + 1, grid_shape[1]-1)

                    if occupancyGrid[i][j] == 100:
                        # print(occupancyGrid[i][j])
                        
                        reward = reward - 50

            if r == 1:
                i = min(i + 1, grid_shape[0]-1)
                next_to_nextstate = i * grid_shape[1] + j
                for r in range(num_actions):
                    # print("SPECIAL CASE !!!")
                    (i,j) = np.unravel_index(next_to_nextstate, occupancyGrid.shape)
                    if r == 0:
                        i = max(i - 1, 0) # manages boundary cases very useful !!
                    if r == 1:
                        i = min(i + 1, grid_shape[0]-1)
                    if r == 2:
                        j = max(j - 1, 0)
                    if r == 3:
                        j = min(j + 1, grid_shape[1]-1)

                    if occupancyGrid[i][j] == 100:
                        # print(occupancyGrid[i][j])
                        
                        reward = reward - 50
            if r == 2:
                j = max(j - 1, 0)
                next_to_nextstate = i * grid_shape[1] + j
                for r in range(num_actions):
                    # print("SPECIAL CASE !!!")
                    (i,j) = np.unravel_index(next_to_nextstate, occupancyGrid.shape)
                    if r == 0:
                        i = max(i - 1, 0) # manages boundary cases very useful !!
                    if r == 1:
                        i = min(i + 1, grid_shape[0]-1)
                    if r == 2:
                        j = max(j - 1, 0)
                    if r == 3:
                        j = min(j + 1, grid_shape[1]-1)

                    if occupancyGrid[i][j] == 100:
                        # print(occupancyGrid[i][j])
                        
                        reward = reward - 50
            if r == 3:
                j = min(j + 1, grid_shape[1]-1)
                next_to_nextstate = i * grid_shape[1] + j
                for r in range(num_actions):
                    # print("SPECIAL CASE !!!")
                    (i,j) = np.unravel_index(next_to_nextstate, occupancyGrid.shape)
                    if r == 0:
                        i = max(i - 1, 0) # manages boundary cases very useful !!
                    if r == 1:
                        i = min(i + 1, grid_shape[0]-1)
                    if r == 2:
                        j = max(j - 1, 0)
                    if r == 3:
                        j = min(j + 1, grid_shape[1]-1)

                    if occupancyGrid[i][j] == 100:
                        # print(occupancyGrid[i][j])
                        
                        reward = reward - 50

            (i,j) = np.unravel_index(next_state, occupancyGrid.shape)
            if occupancyGrid[i][j] == 100:
                # print(occupancyGrid[i][j])
                
                reward = reward - 100
                # print(reward)
            else:
                reward = reward