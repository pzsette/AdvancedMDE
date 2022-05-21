import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance


# Plot clusters
def show_solution(solution, x_label=None):
    plt.scatter(solution.points['x'], solution.points['y'], c=solution.membership_vector, cmap='rainbow')
    plt.xlabel(x_label)
    plt.show()

def print_scolution_score(solution):
    print(f'Solution objective: {solution.get_score()}')


# Compute Euclidean distance between two points
def euclidean_distance(point1, point2):
    '''for value in point:
    point = (x_point, y_point)
    centroid = (x_centroid, y_centroid)'''
    point1 = tuple(point1)
    point2 = tuple(point2)
    return distance.euclidean(point1, point2)


# Build bipartite graph between two solution to start Hungarian algorithm
def build_bipartite_graph(coord_matrix1, coord_matrix2):
    cost_matrix = []
    for index1, point1 in enumerate(coord_matrix1):
        cost_row = []
        for index2, point2 in enumerate(coord_matrix2):
            dst = euclidean_distance(tuple(point1), tuple(point2))
            cost_row.append(dst)
        cost_matrix.append(cost_row)
    return np.asarray(cost_matrix)


# Execute point1 - point2
def subtract_points(point1, point2):
    sub = []
    for value1, value2 in zip(point1, point2):
        sub.append(value1 - value2)
    return sub


# Execute point1 + point2
def sum_points(point1, point2):
    sub = []
    for value1, value2 in zip(point1, point2):
        sub.append(value1 + value2)
    return sub


# Build probabilities vector for roulette wheel function
def build_probabilities_vector(solution, points):
    fitness_vector = []
    for index, point in points.iterrows():
        cluster = solution.membership_vector[index]
        cluster_center = solution.coordinate_matrix[cluster]
        fitness = euclidean_distance(point, cluster_center)
        fitness_vector.append(fitness)
    return fitness_vector


def pr(fitness, total_sum, alpha, n):
    return (1.0 * alpha * fitness / total_sum) + ((1.0 - alpha) / n)