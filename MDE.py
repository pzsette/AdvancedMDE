import random

import numpy as np
from scipy.optimize import linear_sum_assignment

import KMeans
import utils
from Population import Population
from Solution import Solution
from phases.Mutation import Mutator


class MDE:
    def __init__(self,
                 points,
                 n_clusters,
                 population_size=5,
                 f=0.8,
                 max_iteration=10000,
                 max_same_solution_repetition=1000,
                 min_population_diversity=5000
                 ):
        self.points = points
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.f = f
        self.same_solution_repetition = 0
        self.max_iteration = max_iteration
        self.max_same_solution_repetition = max_same_solution_repetition
        self.best_solution = None
        self.min_population_diversity = min_population_diversity
        if self.population_size < 4:
            raise Exception(f'Population size must be 4 or higher')

    def execute_mdo(self):
        p = Population(size=self.population_size, n_clusters=self.n_clusters, points=self.points)
        p.generate_solutions()
        self.best_solution = p.get_best_solution()
        # Loop until stopping one stopping criterion is not satisfied
        while self.check_stopping_criterion(p):
            # Crossover
            for index, solution in enumerate(p.solutions):

                # Select solutions for crossover
                crossover_solution = self.get_crossover_solution(index)

                # Execute crossover

                # Get random solutions
                solution3 = p.get_solution(crossover_solution[0])
                solution2 = p.get_solution(crossover_solution[1])
                solution1 = p.get_solution(crossover_solution[2])
                # Subtraction (S2 - S3)
                sub = self.solutions_subtraction(solution2, solution3)
                # Function F(S2 - S3)
                for i, point in enumerate(sub):
                    for j, value in enumerate(point):
                        sub[i][j] = self.f * value
                # Sum S1 + F(S2 - S3)
                cost_matrix = utils.build_bipartite_graph(sub, solution3.coordinate_matrix)
                matched_points = self.get_matched_points(cost_matrix)
                offspring_coordinate_matrix = []
                for index_centroid in range(self.n_clusters):
                    points_sum = utils.sum_points(solution1.coordinate_matrix[index_centroid],
                                                  sub[matched_points[index_centroid]])
                    offspring_coordinate_matrix.append(points_sum)
                # Build solution
                offspring = Solution(points=self.points, coordinate_matrix=np.asarray(offspring_coordinate_matrix))

                # Mutation
                # TODO: Choose appropriate threshold for mutation execution
                if random.random() > 80:
                    m = Mutator(offspring, self.points)
                    offspring = m.execute_mutation()

                # Local optimization
                offspring = KMeans.compute_solution(self.points, self.n_clusters, start=offspring.coordinate_matrix)
                offspring.update_score()

                # Selection phase
                if offspring.get_score() < p.get_solution(index).get_score():
                    p.replace_solution(index, offspring)
            # print('------------------------------------')

            # print('Computing best solution among population...')
            solution = p.get_best_solution()
            if solution.get_score() == self.best_solution.get_score():
                print('Repetition!')
                self.same_solution_repetition = self.same_solution_repetition + 1
            else:
                print(f'Best solution improved {self.best_solution.get_score()} -> {solution.get_score()}')
                self.best_solution = solution
                self.same_solution_repetition = 0
        return self.best_solution

    def check_stopping_criterion(self, p):
        # Population diversity falls below a threshold
        if p.get_population_diversity() < self.min_population_diversity:
            print('Terminated due to low population diversity!')
            return False
        # Max consecutive iterations performed without any improvement in the best solution
        if self.same_solution_repetition >= self.max_same_solution_repetition:
            print(f'Terminate due to {self.max_same_solution_repetition} best solution repetitions')
            return False
        return True

    def get_crossover_solution(self, index):
        solutions = []
        while len(solutions) < 3:
            selected_index = random.randint(0, self.population_size - 1)
            if selected_index not in solutions and selected_index != index:
                solutions.append(selected_index)
        return solutions

    def solutions_subtraction(self, solution1, solution2):
        cost_matrix = utils.build_bipartite_graph(solution1.coordinate_matrix, solution2.coordinate_matrix)
        col_assignments = self.get_matched_points(cost_matrix)
        new_coordinate_matrix = []
        for index_centroid in range(self.n_clusters):
            subtraction = utils.subtract_points(solution1.coordinate_matrix[index_centroid],
                                                solution2.coordinate_matrix[col_assignments[index_centroid]])
            new_coordinate_matrix.append(subtraction)
        return np.asarray(new_coordinate_matrix)

    def get_matched_points(self, cost_matrix):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return col_ind
