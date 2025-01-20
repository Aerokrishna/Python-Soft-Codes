import numpy as np
import matplotlib.pyplot as plt

# Create 3D axes
ax = plt.axes(projection="3d")

# Generate data for the surface (flat plane in this case)
x = np.arange(0.5, 5, 0.5)  # Array of x-values
y = np.arange(0.5, 4, 0.5)  # Array of y-values
X, Y = np.meshgrid(x, y)  # Create a mesh grid from x and y arrays


Z = 100 * np.ones_like(X)
print(x)
print(Y)

# Plot the surface
ax.plot_surface(X, Y, Z,cmap="Spectral")

# Optional: Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Flat Surface Plot (Z=600)')

plt.show()
