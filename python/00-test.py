import numpy as np
import matplotlib.pyplot as plt

def squared_euclidean_distance(point1, point2):
    return (point1 - point2)**2

def inverse_exp_squared_distance(point1, point2):
    return np.exp(-squared_euclidean_distance(point1, point2))

def generate_line(n):
    return [x/n for x in range(n+1)]

def visualize_inverse_exp_squared_distance(n, point):
"""Visualize the inverse exponential squared distance function for a given point.

    Args:
        n (int): The number of points to generate.
        point (float): The point to compare against.

    Returns:
        None

    Examples:
        >>> visualize_inverse_exp_squared_distance(100, 0.0)

"""
    line = generate_line(n)
    distances = [inverse_exp_squared_distance(p, point) for p in line]
    
    plt.plot(line, distances)
    plt.show()

# Usage:
visualize_inverse_exp_squared_distance(100, 0.0)



import numpy as np
import matplotlib.pyplot as plt

def squared_euclidean_distance(point1, point2):
    print(point1)
    return (point1 - point2)**2

def inverse_exp_squared_distance(point1, point2):
    return np.exp(-squared_euclidean_distance(point1, point2))

def generate_line(distance, step=0.01):
    return np.arange(0, distance+step, step)

def visualize_inverse_exp_squared_distance(distance, point):
    line = generate_line(distance)
    distances = [inverse_exp_squared_distance(p, point) for p in line]
    
    plt.plot(line, distances)
    plt.show()

# Usage:
visualize_inverse_exp_squared_distance(10, 0.0)