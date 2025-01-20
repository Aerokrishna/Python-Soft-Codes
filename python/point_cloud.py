import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

data = {}

def generate_point_cloud(num_points, radius):
    x_points = np.random.uniform(-radius,radius,num_points) # gets num_points number of points between 0 to radius
    y_points = np.random.uniform(-radius,radius,num_points)
    z_points = np.random.uniform(-radius,radius,num_points)

    for i in range(num_points):
        x = x_points[i]
        y = y_points[i]
        z = z_points[i]

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan(y/x)
        phi = np.arccos(z/r)

        data.update({r:(theta,phi)})
    
    keys = list(data.keys())
    
    for i in range(len(keys)):
        if keys[i] < 3 and keys[i-1] > 5:
            print("HUMMUS")

    return np.column_stack((x_points,y_points,z_points)) # stacks one d arrays as columns and returns a 2d matrix

def plot_points(num_points=10,radius=2):
    '''
    Visualize the point cloud, robot, goal, path
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    points = generate_point_cloud(num_points = num_points,radius = radius)
    print(points)
    # Plot the data points
    for i in range(num_points):
        point = points[i]
        ax.scatter(point[0], point[1], point[2], c='blue', marker='o')  # Customize color and marker as desired

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set axes limits (optional, adjust based on your data)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)

    # Add a title (optional)
    plt.title('3D Scatter Plot')

    plt.show()

plot_points(num_points=100,radius=1)


