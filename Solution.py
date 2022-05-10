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
        for index, row in self.points.iterrows():
            cluster = self.membership_vector[index]
            cluster_center = self.coordinate_matrix[cluster]
            sum += utils.euclidean_distance(row['x'], row['y'], cluster_center[0], cluster_center[1])
        return sum

    def get_score(self):
        return self._score

    def get_memb_vect_from_coord_matrix(self):
        membership_vector = []
        for row in self.points.iterrows():
            assigned_centroid = 0
            x_point = float(row[1][0])
            y_point = float(row[1][1])
            min_dst = utils.euclidean_distance(x_point, y_point, self.coordinate_matrix[0][0], self.coordinate_matrix[0][1])
            for index, centroid in enumerate(self.coordinate_matrix):
                dst = utils.euclidean_distance(x_point, y_point, centroid[0], centroid[1])
                if dst < min_dst:
                    assigned_centroid = index
                    min_dst = dst
            membership_vector.append(assigned_centroid)
        return np.asarray(membership_vector)

