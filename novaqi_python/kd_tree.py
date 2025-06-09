from scipy.spatial import KDTree
import numpy as np

# Your roadmap sample points
sample_x = [1.0, 2.0, 3.0, 5.0]
sample_y = [1.0, 4.0, 2.0, 1.0]

# Combine into a 2D array of shape (N, 2)
roadmap_points = np.vstack((sample_x, sample_y)).T

# Build the KDTree
roadmap_kd_tree = KDTree(roadmap_points)

rx, ry = 2.1, 3.8  # Robot's candidate point

# return the nearest neighbor to the point (rx, ry)
distance, index = roadmap_kd_tree.query([rx, ry], k=1)# k means how many nearest neighbor you want

# return all points within a radius of 1.0 from the point (rx, ry)
indices = roadmap_kd_tree.query_ball_point([rx, ry], r=1.0) 