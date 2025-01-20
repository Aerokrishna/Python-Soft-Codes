import matplotlib.pyplot as plt
import numpy as np
import csv

# Path to your occupancy grid CSV file
matrix_path = '/home/krishnapranav/novaqi_python/real_map4_Ogrid.csv'

# Load the occupancy grid from the CSV file
with open(matrix_path, 'r') as f:
    reader = csv.reader(f)
    loaded_matrix = [list(map(int, row)) for row in reader]  # Ensure integers for 0 and 1

occupancy_grid = np.array(loaded_matrix)

def plot_occupancy_grid(grid, save_path=None):
    plt.figure(figsize=(40, 40))  # Create a figure
    cmap = plt.cm.binary  # Binary colormap (black and white)
    plt.imshow(grid, cmap=cmap, origin='upper')  # Plot the grid
    plt.title("Occupancy Grid Map")
    plt.xlabel("X-axis")  # Customize axis labels as needed
    plt.ylabel("Y-axis")
    plt.colorbar(label="Occupancy (0 = Free, 1 = Occupied)")  # Add a color bar for clarity
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the figure
        print(f"Plot saved to {save_path}")
    
    plt.show()  # Display the figure

# Example usage
plot_occupancy_grid(occupancy_grid, save_path="occupancy_grid_map.png")
