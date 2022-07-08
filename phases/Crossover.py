import numpy as np
from scipy.optimize import linear_sum_assignment
import utils
from matching import ExactMatching, GreedyMatching
from models import Population
from matching.ExactMatching import exact_matching
from matching.GreedyMatching import greedy_matching
from models.Solution import Solution


def solutions_subtraction(solution1, solution2):
    # if matching == 'exact':
    #     col_assignments = exact_matching(solution1.coordinate_matrix, solution2.coordinate_matrix)
    # elif matching == 'greedy':
    #     col_assignments = greedy_matching(solution1.coordinate_matrix, solution2.coordinate_matrix)
    # else:
    #     raise ValueError('Error in matching algorithm name')
    #
    # print("Matching vector", col_assignments)
    new_coordinate_matrix = []
    for index_centroid in range(len(solution1.coordinate_matrix)):
        new_coordinate_matrix.append(utils.subtract_points(solution1.coordinate_matrix[index_centroid],
                                                           solution2.coordinate_matrix[index_centroid]))
    return new_coordinate_matrix


def solution_sum(coordinate_matrix, solution3):
    # if matching == 'exact':
    #     matched_points = exact_matching(coordinate_matrix, solution3.coordinate_matrix)
    # elif matching == 'greedy':
    #     matched_points = greedy_matching(coordinate_matrix, solution3.coordinate_matrix)
    # else:
    #     raise ValueError('Error in matching algorithm name')
    offspring_coordinate_matrix = []
    for index_centroid in range(len(coordinate_matrix)):
        offspring_coordinate_matrix.append(utils.sum_points(solution3.coordinate_matrix[index_centroid],
                                                            coordinate_matrix[index_centroid]))

    return offspring_coordinate_matrix


class Crossover:

    def __init__(self, population, index1, index2, index3, matching_type):
        self.population = population
        self.index1 = index1
        self.index2 = index2
        self.index3 = index3
        self.matching_type = matching_type

    def execute_crossover(self):
        # Execute crossover
        self.matching(self.index3, self.index2)
        solution3 = self.population.get_solution(self.index3)
        solution2 = self.population.get_solution(self.index2)
        # Subtraction (S2 - S3)
        sub = solutions_subtraction(solution2, solution3)
        self.matching(self.index3, self.index1)
        # Function F(S2 - S3)
        rnd = np.random.uniform(low=0.5, high=0.8)
        m = len(sub)
        for i in range(m):
            for j in range(len(sub[i])):
                sub[i][j] *= rnd
        # Sum S1 + F(S2 - S3)
        solution1 = self.population.get_solution(self.index1)
        sum_result = solution_sum(sub, solution1)
        # Build solution
        return sum_result

    def matching(self, index1, index2):
        solution1 = self.population.get_solution(index1)
        solution2 = self.population.get_solution(index2)

        if self.matching_type == 'exact':
            matched_points = ExactMatching.exact_matching(solution1.coordinate_matrix, solution2.coordinate_matrix)
        elif self.matching_type == 'greedy':
            matched_points = GreedyMatching.greedy_matching(solution1.coordinate_matrix, solution2.coordinate_matrix)
        else:
            raise Exception('Error in matching type')

        # print(matched_points)

        new_centroid = []
        for index in range(len(solution1.coordinate_matrix)):
            new_centroid.append(solution2.coordinate_matrix[matched_points[index]])

        new_solution = Solution(points=self.population.points, coordinate_matrix=new_centroid, score=solution2.get_score())

        self.population.replace_solution(index2, new_solution)




