import numpy as np

coords = np.zeros((9,2))

for i in range(9):
    grid_x, grid_y = np.unravel_index(i,(3,3))
    coords[i] = np.array([grid_x,grid_y])
print(coords)
