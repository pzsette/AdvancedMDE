from random import randrange
from numba import jit

import numpy as np
from scipy.spatial import distance


# Compute Euclidean distance between two points
def euclidean_distance(point1, point2):
    point1 = tuple(point1)
    point2 = tuple(point2)
    return distance.euclidean(point1, point2)


# Build bipartite graph between two solution to start Hungarian algorithm
#@jit
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
@jit
def subtract_points(point1, point2):
    sub = []
    for value1, value2 in zip(point1, point2):
        sub.append(value1 - value2)
    return sub


# Execute point1 + point2
@jit
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


def get_random_in_range_less_one(upper_bound, excluded):
    index = randrange(upper_bound)
    while index == excluded:
        index = randrange(4)
    return index


def print_result(score, calls):
    print(f'  Best score -> {score}')
    print(f'  Number of K-MEANS executions -> {calls}')