from scipy.optimize import linear_sum_assignment

import utils


def get_matched_points(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind


def exact_matching(sol1_coordinate_matrix, sol2_coordinate_matrix):
    cost_matrix = utils.build_bipartite_graph(sol1_coordinate_matrix, sol2_coordinate_matrix)
    return get_matched_points(cost_matrix)
