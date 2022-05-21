import math
import random
from random import randrange

import numpy as np

import utils


def get_close_centroid_index(point, coordinate_matrix):
    min_dst = None
    best = None
    for i in range(0, len(coordinate_matrix)):
        distance = utils.euclidean_distance(point, tuple(coordinate_matrix[i]))
        if min_dst is None or distance < min_dst:
            min_dst = distance
            best = i
    return best


def rebuild_membership_vector(points, coordinate_matrix):
    new_membership_vector = []
    for _, point in points.iterrows():
        new_membership_vector.append(get_close_centroid_index(tuple(point), coordinate_matrix))
    return np.asarray(new_membership_vector)


def find_index(values, key, first, last):
    if values[first] <= key <= values[first + 1]:
        return first

    imid = first + math.ceil((last - first)/2)

    if first == last or imid == last:
        return -1

    if values[imid] > key:
        return find_index(values, key, first, imid)
    else:
        return find_index(values, key, imid, last)


def get_roulette_index(solution, points):
    n = len(points)
    fitness_vector = utils.build_probabilities_vector(solution, points)
    fitness_sum = sum(fitness_vector)

    # Build random wheel vector
    alpha = 5
    pr = []
    pr.append(utils.pr(fitness_vector[0], fitness_sum, alpha, n))
    for i in range(1, len(fitness_vector)):
        pr.append(pr[i-1] + utils.pr(fitness_vector[0], fitness_sum, alpha, n))

    r = random.uniform(0, pr[-1])

    return find_index(pr, r, 0, len(fitness_vector)-1)


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

        # Select new random point
        index_new_centroid = get_roulette_index(self.solution, self.points)
        new_centroid = self.points.iloc[index_new_centroid]
        self.solution.coordinate_matrix = np.vstack([self.solution.coordinate_matrix, new_centroid])

        # Rebuild membership vector with new point
        self.solution.membership_vector = rebuild_membership_vector(self.points, self.solution.coordinate_matrix)
        self.solution.compute_score()
        return self.solution
