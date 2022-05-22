import sys

import utils


def greedy_matching(sol1_coordinate_matrix, sol2_coordinate_matrix):
    assigned = []
    for point1 in sol1_coordinate_matrix:
        min_distance = sys.float_info.max
        min_index = -1
        for index, point2 in enumerate(sol2_coordinate_matrix):
            if index not in assigned:
                distance = utils.euclidean_distance(tuple(point1), tuple(point2))
                if distance < min_distance:
                    min_distance = distance
                    min_index = index
        if min_index == -1:
            raise ValueError('Every point of solution1 must be assigned to one point of solution2')
        assigned.append(min_index)
    return assigned
