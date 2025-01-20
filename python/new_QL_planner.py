import matplotlib.pyplot as plt
import numpy as np

X_grid = np.tile(np.arange(40),(40,1))
Y_grid = np.tile(np.arange(40).reshape((40,1)),(1,40))
Z_grid = np.full((40,40),20)
ax = plt.axes(projection='3d')
ax.plot_surface(X_grid,Y_grid,Z_grid,alpha=0.7)

X_robot = np.tile(np.arange(6),(6,1))
Y_robot = np.tile(np.arange(6).reshape((6,1)),(1,6))
# X_robot,Y_robot = np.meshgrid(X_robot,Y_robot)
Z_robot = np.full((6,6),20)

ax.plot_surface(X_robot,Y_robot,Z_robot,color='r')
plt.show()