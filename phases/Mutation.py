import math
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


def get_roulette_index(coordinate_matrix, points):
    n = len(points)
    fitness_vector = utils.build_probabilities_vector(coordinate_matrix=np.array(coordinate_matrix), points=points)
    # print(fitness_vector)
    fitness_sum = sum(fitness_vector)

    # Build random wheel vector
    alpha = 0.5
    pr = [utils.pr(fitness_vector[0], fitness_sum, alpha, n)]
    for i in range(1, len(fitness_vector)):
        pr.append(pr[i-1] + utils.pr(fitness_vector[i], fitness_sum, alpha, n))

    r = np.random.uniform(0, pr[-1])

    return find_index(pr, r, 0, len(fitness_vector)-1)


class Mutator:
    def __init__(self, solution):
        self.solution = solution

    def execute_mutation(self):
        # Select random solution to delete
        index_to_delete = np.random.randint(low=0, high=len(self.solution.coordinate_matrix))
        # Remove random centroid
        self.solution.remove_center(index_to_delete)

        # Select new point through roulette wheel function
        index_new_centroid = get_roulette_index(self.solution.coordinate_matrix, self.solution.points)
        self.solution.reinsert_center(point_index=index_new_centroid, index_to_replace=index_to_delete)
