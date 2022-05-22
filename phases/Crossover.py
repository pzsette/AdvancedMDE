import random

import numpy as np
from scipy.optimize import linear_sum_assignment

import utils
from Solution import Solution

from matching.ExactMatching import exact_matching
from matching.GreedyMatching import greedy_matching


def get_matched_points(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind


def solutions_subtraction(solution1, solution2, matching):
    if matching == 'exact':
        col_assignments = exact_matching(solution1.coordinate_matrix, solution2.coordinate_matrix)
    elif matching == 'greedy':
        col_assignments = greedy_matching(solution1.coordinate_matrix, solution2.coordinate_matrix)
    else:
        raise ValueError('Error in matching algorithm name')
    new_coordinate_matrix = []
    for index_centroid in range(len(solution1.coordinate_matrix)):
        subtraction = utils.subtract_points(solution1.coordinate_matrix[index_centroid],
                                            solution2.coordinate_matrix[col_assignments[index_centroid]])
        new_coordinate_matrix.append(subtraction)
    return np.asarray(new_coordinate_matrix)


def solution_sum(coordinate_matrix, solution3, matching):
    if matching == 'exact':
        matched_points = exact_matching(coordinate_matrix, solution3.coordinate_matrix)
    elif matching == 'greedy':
        matched_points = greedy_matching(coordinate_matrix, solution3.coordinate_matrix)
    else:
        raise ValueError('Error in matching algorithm name')
    offspring_coordinate_matrix = []
    for index_centroid in range(len(coordinate_matrix)):
        points_sum = utils.sum_points(solution3.coordinate_matrix[index_centroid],
                                      coordinate_matrix[matched_points[index_centroid]])
        offspring_coordinate_matrix.append(points_sum)
    return np.asarray(offspring_coordinate_matrix)


class Crossover:
    @staticmethod
    def execute_crossover(points, solution1, solution2, solution3, matching):
        # Execute crossover
        # Subtraction (S2 - S3)
        sub = solutions_subtraction(solution2, solution3, matching)
        # Function F(S2 - S3)
        for i, point in enumerate(sub):
            for j, value in enumerate(point):
                sub[i][j] = random.uniform(0.5, 0.8) * value
        # Sum S1 + F(S2 - S3)
        sum_result = solution_sum(sub, solution1, matching)
        # Build solution
        return Solution(points=points, coordinate_matrix=sum_result)
