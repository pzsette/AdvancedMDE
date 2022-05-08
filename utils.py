import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance


# Plot clusters
def show_solution(points, solution):
    plt.scatter(points['x'], points['y'], c=solution.membership_vector, cmap='rainbow')
    plt.show()


# Compute Euclidean distance between two points
def euclidean_distance(x_point, y_point, x_centroid, y_centroid):
    point = (x_point, y_point)
    centroid = (x_centroid, y_centroid)
    return distance.euclidean(point, centroid)


# Build bipartite graph between two solution to start Hungarian algorithm
def build_bipartite_graph(coord_matrix1, coord_matrix2):
    cost_matrix = np.empty(shape=(len(coord_matrix1), len(coord_matrix2)))
    for point1 in coord_matrix1:
        cost_row = np.empty(shape=(len(coord_matrix1)))
        for point2 in coord_matrix2:
            np.append(cost_row, euclidean_distance(point1[0], point1[1], point2[0], point2[1]))
        np.append(cost_matrix, cost_row)
    return cost_matrix
