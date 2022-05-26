import math
import sys

import numpy as np

import utils

def get_memb_vect_from_coord_matrix(points, coord_matrix):
    membership_vector = np.empty(shape=0)
    for (_, row) in points.iterrows():
        assigned_centroid = 0
        min_dst = sys.float_info.max

        for index, centroid in enumerate(coord_matrix):
            dst = utils.euclidean_distance(tuple(row), coord_matrix[index])
            if dst < min_dst:
                assigned_centroid = index
                min_dst = dst
        membership_vector = np.append(membership_vector, assigned_centroid)
    return np.asarray(membership_vector)


class Solution:
    def __init__(self, points, coordinate_matrix, score, membership_vector=None):
        self.coordinate_matrix = coordinate_matrix
        self.points = points
        self.membership_vector = membership_vector
        self._score = score

    def compute_score(self):
        if self.membership_vector is None:
            self.membership_vector = get_memb_vect_from_coord_matrix(self.points, self.coordinate_matrix)
        sum = 0
        for index, point in self.points.iterrows():
            cluster = self.membership_vector[index]
            cluster_center = self.coordinate_matrix[int(cluster)]
            sum += math.pow(utils.euclidean_distance(point, cluster_center), 2)
        return sum

    def get_score(self):
        return self._score

    def update_score(self):
        self._score = self.compute_score()