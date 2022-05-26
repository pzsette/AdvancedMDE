import numpy as np

import random
from matching.ExactMatching import exact_matching


def greedy_generation(solution, solution_to_compare, f):
    phi = 1 if solution.get_score() > solution_to_compare.get_score() else -1
    generated_solution = []
    col_assignments = exact_matching(solution.coordinate_matrix, solution_to_compare.coordinate_matrix)
    for index_centroid in range(len(solution.coordinate_matrix)):
        if random.uniform(0, 1) < 0.7:
            # F(p_j - p_i)
            subtraction = subtract_points_and_scale(
                solution_to_compare.coordinate_matrix[col_assignments[index_centroid]],
                solution.coordinate_matrix[index_centroid],
                f)
            # p_i + phi * F(p_j - p_i)
            generated_point = []
            for value1, value2 in zip(solution.coordinate_matrix[index_centroid], subtraction):
                generated_point.append(value1 + phi*value2)
            generated_solution.append(generated_point)
        else:
            generated_solution.append(solution.coordinate_matrix[index_centroid])
    return np.asarray(generated_solution)


# Execute point1 - point2 and scale by f
def subtract_points_and_scale(point1, point2, f):
    #print('point 1,2')
    #print(point1)
    #print(point2)
    #print(f)
    sub = []
    for value1, value2 in zip(point1, point2):
        sub.append(f*(abs(value1 - value2)))
    #print('sub')
    #print(sub)
    return sub








