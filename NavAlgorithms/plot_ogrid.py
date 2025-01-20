import numpy as np
import matplotlib.pyplot as plt

# Set grid size
grid_size = 10

# Randomly fill each cell as either free (0) or occupied (1)
occupancy_grid = np.random.choice([0, 1], size=(grid_size, grid_size), p=[0.5, 0.5])

# Plot the grid
plt.figure(figsize=(6, 6))
# plt.imshow(occupancy_grid, cmap="gray_r")
plt.xticks(range(grid_size))  # Remove x-axis numbers
plt.yticks(range(grid_size))  # Remove y-axis numbers
plt.gca().set_aspect('equal')  # Ensure cells are square
plt.grid(color='black', linestyle='-', linewidth=1)
plt.title("10x10 Occupancy Grid with Random Obstacles")

# Add red dots only on free cells, centered within each cell
for i in range(grid_size):
    for j in range(grid_size):
        plt.scatter(j + 0.5, i + 0.5, color="red", s=50)  # Center dot in free cell

# Save the plot
plt.savefig("occupancy_grid_with_dots.png", bbox_inches='tight', dpi=300)
plt.show()
