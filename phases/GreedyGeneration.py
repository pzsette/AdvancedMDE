import numpy as np

from matching.GreedyMatching import greedy_matching
from matching.ExactMatching import exact_matching


def greedy_generation(solution, solution_to_compare, f, matching_type):
    phi = 1 if solution.get_score() > solution_to_compare.get_score() else -1
    generated_solution = []
    if matching_type == 'exact':
        col_assignments = exact_matching(solution.coordinate_matrix, solution_to_compare.coordinate_matrix)
    elif matching_type == 'greedy':
        col_assignments = greedy_matching(solution.coordinate_matrix, solution_to_compare.coordinate_matrix)
    else:
        raise Exception('Error in matching type')

    for index_centroid in range(len(solution.coordinate_matrix)):
        if np.random.uniform(0, 1) < 1:
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
    return phi, np.asarray(generated_solution)


# Execute point1 - point2 and scale by f
def subtract_points_and_scale(point1, point2, f):
    sub = []
    for value1, value2 in zip(point1, point2):
        sub.append(f*(value1 - value2))
    return sub








