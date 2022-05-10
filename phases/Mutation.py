from random import randrange

import numpy as np


class Mutator:
    def __init__(self, solution, points):
        self.solution = solution
        self.points = points

    def execute_mutation(self):
        # Select random solution to delete
        print(self.solution.coordinate_matrix)
        index_to_delete = randrange(0, len(self.solution.coordinate_matrix))
        print(index_to_delete)
        self.solution.coordinate_matrix = np.delete(np.asmatrix(self.solution.coordinate_matrix), index_to_delete, axis=0)
        print(self.solution.coordinate_matrix)