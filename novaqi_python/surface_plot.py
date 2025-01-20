import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np

matrix_path = '/home/krishnapranav/novaqi_python/bot_0_real_map4_gridTable.csv'
grid_shape = [40, 40]
plot_arrayY = np.tile(np.arange(grid_shape[1]), (grid_shape[0], 1))
plot_arrayX = np.tile(np.arange(grid_shape[0]).reshape(grid_shape[0], 1), (1, grid_shape[1]))
        
with open(matrix_path, 'r') as f:
    reader = csv.reader(f)
    loaded_matrix = [list(map(float, row)) for row in reader]

gridMap_optimal_values = np.array(loaded_matrix)

def plot_grid(X, Y, Z, save_path):
    fig = plt.figure(figsize=(8, 8))  # Create a figure object
    ax = fig.add_subplot(111, projection="3d")  # Add 3D subplot to the figure
    ax.plot_surface(X, Y, Z, cmap="plasma")
    ax.set_title("OFFLINE LEARNING PLOT")
    ax.set_xlabel('Y-axis')  # Customize these labels as needed
    ax.set_ylabel('X-axis')
    
    # Save the figure before showing it
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    # Now show the figure
    plt.show()

plot_grid(X=plot_arrayX, Y=plot_arrayY, Z=gridMap_optimal_values, save_path="heatmap3d_plot.png")
