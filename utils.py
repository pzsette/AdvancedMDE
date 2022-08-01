from numba import jit

import numpy as np
from scipy.spatial import distance
import utils


def dist(x, y):
    subs = []
    for x_comp, y_comp in zip(x, y):
        sub = x_comp - y_comp
        sub_sqrt = sub**2
        subs.append(sub_sqrt)
    sum = np.sum(subs)
    sqrt = np.sqrt(sum)
    return sqrt


# Compute Euclidean distance between two points
def euclidean_distance(point1, point2):
    return distance.euclidean(point1, point2)


# Build bipartite graph between two solution to start Hungarian algorithm
def build_bipartite_graph(coord_matrix1, coord_matrix2):
    cost_matrix = []
    for index1, point1 in enumerate(coord_matrix1):
        cost_row = []
        for index2, point2 in enumerate(coord_matrix2):
            dst = euclidean_distance(point1, point2)**2
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


@jit(nopython=True)
def get_memb_vect_from_coord_matrix(points, coord_matrix):
    membership_vector = []
    for row in points:
        min_dst = np.linalg.norm(row - coord_matrix[0])
        assigned_centroid = 0
        for index, centroid in enumerate(coord_matrix):
            dst = np.linalg.norm(row - coord_matrix[index])
            if dst < min_dst:
                assigned_centroid = index
                min_dst = dst
        membership_vector.append(assigned_centroid)
    return np.asarray(membership_vector)


# Build probabilities vector for roulette wheel function
def build_probabilities_vector(coordinate_matrix, points):
    fitness_vector = []
    membership_vector = utils.get_memb_vect_from_coord_matrix(points, coordinate_matrix)
    for index, point in enumerate(points):
        cluster = membership_vector[index]
        cluster_center = coordinate_matrix[cluster]
        fitness = euclidean_distance(point, cluster_center)
        fitness_vector.append(fitness)
    return fitness_vector


def pr(fitness, total_sum, alpha, n):
    return (alpha * fitness / total_sum) + ((1.0 - alpha) / n)


def get_random_in_range_less_one(upper_bound, excluded):
    index = np.random.randint(low=0, high=upper_bound)
    while index == excluded:
        index = np.random.randint(low=0, high=upper_bound)
    return index


def print_result(score, calls, time):
    print(f'  Solution objective -> {score}')
    print(f'  CPU time (s): {time}')
    print(f'  Number of K-MEANS executions -> {calls}')


def print_summary_results(result_matrix, clusters):
    matrix = np.matrix(result_matrix)
    print('\n')
    print('########### RESUME ###########')
    for clust_in, cluster in enumerate(clusters):
        print(f'{cluster} CLUSTER')
        print(f'  Best score -> {np.min(matrix[clust_in*3,:])}')
        print(f'  Avg K-Means calls -> {np.mean(matrix[clust_in * 3 +1, :])}')
        print(f'  Avg exec time -> {np.mean(matrix[clust_in * 3 + 2, :])}')
    print('\n')


def has_to_be_repaired(solution, m):
    sizes = [0] * m
    empty_clusters = []

    for assignment in solution.get_membership_vector():
        sizes[assignment] += 1
    for i in range(m):
        if sizes[i] == 0:
            empty_clusters.append(i)
    if len(empty_clusters) > 0:
        return True
    else:
        return False
