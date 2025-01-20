import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_distribution(mean, std_dev, num_points=1000):
    """
    Generate a Gaussian distribution with given mean and standard deviation using NumPy's inbuilt function.

    Parameters:
    mean (float): The mean of the Gaussian distribution.
    std_dev (float): The standard deviation of the Gaussian distribution.
    num_points (int): The number of points to generate. Default is 1000.

    Returns:
    data (numpy.ndarray): The generated Gaussian distribution data.
    """
    data = np.random.normal(loc=mean, scale=std_dev, size=num_points)
    return data

# Example usage:
mean = 5
std_dev = 1
num_points = 100
data = generate_gaussian_distribution(mean, std_dev, num_points)

# Plotting the Gaussian distribution
plt.hist(data, bins=100, density=True, alpha=0.6, color='g', edgecolor='black')

plt.title(f'Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
