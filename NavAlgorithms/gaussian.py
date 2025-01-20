import numpy as np

def gaussian_distribution(mean, std_dev, num_points=25):
    """
    Generate a Gaussian distribution with given mean and standard deviation.

    Parameters:
    mean (float): The mean of the Gaussian distribution.7
    std_dev (float): The standard deviation of the Gaussian distribution.
    num_points (int): The number of points to generate. Default is 1000.

    Returns:
    x (numpy.ndarray): The x values.
    y (numpy.ndarray): The corresponding y values of the Gaussian distribution.
    """
    x = np.linspace(mean - 6*std_dev, mean + 6*std_dev, num_points)
    y = (1 / (np.sqrt(2 * np.pi * std_dev**2))) * np.exp(- (x - mean)**2 / (2 * std_dev**2))
    amp = (1 / (np.sqrt(2 * np.pi * std_dev**2)))
    return x, y,amp

# Example usage:
mean = 9
std_dev = 1.0
x, y,amp = gaussian_distribution(mean, std_dev)
print(x)
print(y)
print(amp)
import matplotlib.pyplot as plt

# plt.plot(x, y)

plt.bar(x,y)
plt.title(f'Gaussian Distribution (mean={mean}, std_dev={std_dev})')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
