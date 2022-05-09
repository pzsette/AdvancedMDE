import utils


class Solution:
    def __init__(self, membership_vector, coordinate_matrix):
        self.membership_vector = membership_vector
        self.coordinate_matrix = coordinate_matrix
        self._score = None

    def get_score(self, points):
        if self._score is None:
            sum = 0
            for index, row in points.iterrows():
                cluster = self.membership_vector[index]
                cluster_center = self.coordinate_matrix[cluster]
                sum += utils.euclidean_distance(row['x'], row['y'], cluster_center[0], cluster_center[1])
            self._score = sum
        return self._score

