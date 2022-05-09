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
    cost_matrix = []
    for point1 in coord_matrix1:
        cost_row = []
        for point2 in coord_matrix2:
            dst = euclidean_distance(point1[0], point1[1], point2[0], point2[1])
            cost_row.append(dst)
        cost_matrix.append(cost_row)
    return np.asarray(cost_matrix)


# Execute point1 - point2
def subtract_points(point1, point2):
    x = point1[0] - point2[0]
    y = point1[1] - point2[1]
    return [x, y]


# Execute point1 - point2
def sum_points(point1, point2):
    x = point1[0] + point2[0]
    y = point1[1] + point2[1]
    return [x, y]
