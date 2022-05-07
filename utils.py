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


# Compute sum of square on a solution
def sum_of_square(points, solution):
    sum = 0
    for index, row in points.iterrows():
        cluster = solution.membership_vector[index]
        cluster_center = solution.coordinate_matrix[cluster]
        sum += euclidean_distance(row['x'], row['y'], cluster_center[0], cluster_center[1])
    return sum


# Build bipartite graph between two solution to start Hungarian algorithm
def build_bipartite_graph(solution1, solution2):
    cost_matrix = np.expty()
    for point1 in solution1.coordinate_matrix:
        cost_row = np.empty()
        for point2 in solution2.coordinate_matrix:
            cost_row.append(euclidean_distance(point1[0], point1[1], point2[0], point2[1]))
        cost_matrix.append(cost_row)
    return cost_matrix
