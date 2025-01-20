import numpy as np
import matplotlib.pyplot as plt

def cubic_bezier(t, p0, p1, p2, p3):
    return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

def plot_cubic_bezier(p0, p1, p2, p3, num_points=10):
    t_values = np.linspace(0, 1, num_points)
    x_values = []
    y_values = []
    for t in t_values:
        x, y = cubic_bezier(t, p0, p1, p2, p3)
        x_values.append(x)
        y_values.append(y)
    print("x",x_values)
    print("y",y_values)
    plt.plot(x_values, y_values, label='Original Curve')

# Initial control points
p0 = np.array([0, 0])
p1 = np.array([1, 3])
p2 = np.array([4, -1])
p3 = np.array([5, 2])

# Plotting original curve
plot_cubic_bezier(p0, p1, p2, p3)

# Manipulating control points
p1_new = np.array([2, 4])
p2_new = np.array([3, 1])

# Plotting manipulated curve
plot_cubic_bezier(p0, p1_new, p2_new, p3)
plt.scatter([p0[0], p1_new[0], p2_new[0], p3[0]], [p0[1], p1_new[1], p2_new[1], p3[1]], color='red', label='Manipulated Control Points')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Manipulating Control Points in Cubic Bezier Curve')
plt.grid(True)
plt.show()