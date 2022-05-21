import math

import numpy as np

import utils


class Solution:
    def __init__(self, points, coordinate_matrix, membership_vector=None):
        self.coordinate_matrix = coordinate_matrix
        self.points = points
        if membership_vector is None:
            self.membership_vector = self.get_memb_vect_from_coord_matrix()
        else:
            self.membership_vector = membership_vector
        self._score = self.compute_score()

    def compute_score(self):
        sum = 0
        for index, point in self.points.iterrows():
            cluster = self.membership_vector[index]
            cluster_center = self.coordinate_matrix[cluster]
            sum += math.pow(utils.euclidean_distance(point, cluster_center), 2)
        return sum

    def get_score(self):
        return self._score

    def update_score(self):
        self._score = self.compute_score()

    def get_memb_vect_from_coord_matrix(self):
        membership_vector = []
        for (_, row) in self.points.iterrows():
            assigned_centroid = 0
            point = tuple(row)
            min_dst = utils.euclidean_distance(point, self.coordinate_matrix[0])
            for index, centroid in enumerate(self.coordinate_matrix):
                dst = utils.euclidean_distance(point, self.coordinate_matrix[index])
                if dst < min_dst:
                    assigned_centroid = index
                    min_dst = dst
            membership_vector.append(assigned_centroid)
        return np.asarray(membership_vector)

