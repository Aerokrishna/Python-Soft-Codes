import matplotlib.pyplot as plt
import csv
import numpy as np
import os

# Path to your CSV file
matrix_path = '/home/krishnapranav/novaqi_python/bot_0_real_map4_gridTable.csv'

# Define grid shape
grid_shape = [40, 40]

# Load the matrix from the CSV file
with open(matrix_path, 'r') as f:
    reader = csv.reader(f)
    loaded_matrix = [list(map(float, row)) for row in reader]

gridMap_optimal_values = np.array(loaded_matrix)

def plot_heatmap(Z, save_path):
    plt.figure(figsize=(8, 8))  # Adjust figure size as needed
    plt.imshow(Z, cmap='plasma', origin='upper')
    plt.colorbar(label="Value")  # Add a color bar to indicate the scale
    plt.title("OFFLINE LEARNING HEATMAP")
    plt.xlabel('Y-axis')  # Customize these labels as needed
    plt.ylabel('X-axis')
    
    # Save the figure before showing it
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    # Now show the figure
    plt.show()

# Example usage
plot_heatmap(gridMap_optimal_values, save_path="heatmap.png")
