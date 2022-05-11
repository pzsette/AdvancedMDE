from random import randrange

import numpy as np

import utils


def get_close_centroid_index(x, y, coordinate_matrix):
    min_dst = None
    best = None
    for i in range(0, len(coordinate_matrix)):
        x_cluster = coordinate_matrix[i][0]
        y_cluster = coordinate_matrix[i][1]
        distance = utils.euclidean_distance(x, y, x_cluster, y_cluster)
        if min_dst is None or distance < min_dst:
            min_dst = distance
            best = i
    return best


def rebuild_membership_vector(points, coordinate_matrix):
    new_membership_vector = []
    for index, row in points.iterrows():
        x = row['x']
        y = row['y']
        new_membership_vector.append(get_close_centroid_index(x, y, coordinate_matrix))
    return np.asarray(new_membership_vector)


def get_roulette_index(size):
    # TODO: Implement roulette function
    return randrange(0, size)


class Mutator:
    def __init__(self, solution, points):
        self.solution = solution
        self.points = points

    def execute_mutation(self):
        # Select random solution to delete
        index_to_delete = randrange(0, len(self.solution.coordinate_matrix))
        self.solution.coordinate_matrix = np.delete(np.asarray(self.solution.coordinate_matrix), index_to_delete, axis=0)

        # Rebuild membership vector
        self.solution.membership_vector = rebuild_membership_vector(self.points, self.solution.coordinate_matrix)
        utils.show_solution(self.solution)

        # Select new random point
        index_new_centroid = get_roulette_index(len(self.points))
        new_centroid = self.points.iloc[index_new_centroid]
        self.solution.coordinate_matrix = np.vstack([self.solution.coordinate_matrix,
                                                     np.asarray([new_centroid['x'], new_centroid['y']])])

        # Rebuild membership vector with new point
        self.solution.membership_vector = rebuild_membership_vector(self.points, self.solution.coordinate_matrix)
        self.solution.compute_score()
        return self.solution
