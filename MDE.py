import random

import numpy as np
from scipy.optimize import linear_sum_assignment

import KMeans
import utils


class MDE:
    def __init__(self, points, n_clusters, population_size=10, f=0.5):
        self.points = points
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.f = f

    def execute_mdo(self):
        population = []
        # Init population with K-random execution of K-means
        for i in range(self.population_size):
            population.append(KMeans.compute_solution(self.points, self.n_clusters))

        for index, solution in enumerate(population):
            print(f'index {index}')
            print(self.get_crossover_solution(index))

        # Loop until stopping one stopping criterion is not satisfied
        while self.check_stopping_criterion():

            # Crossover
            for index, solution in enumerate(population):
                # Select solutions for crossover
                crossover_solution = self.get_crossover_solution(index)
                # Execute crossover
                solution0 = population[crossover_solution[0]]
                solution1 = population[crossover_solution[1]]
                solution2 = population[crossover_solution[2]]
                sub = self.solutions_subtraction(solution0, solution1)




        # Local optimization
        solution = KMeans.compute_solution(self.points, self.n_clusters, start=population[0].coordinate_matrix)
        return solution

    def check_stopping_criterion(self):
        return True

    def get_crossover_solution(self, index):
        solutions = []
        while len(solutions) < 3:
            selected_index = random.randint(0, self.population_size-1)
            if selected_index not in solutions and selected_index != index:
                solutions.append(selected_index)
        return solutions

    def solutions_subtraction(self, solution1, solution2):
        cost_matrix = utils.build_bipartite_graph(solution1, solution2)
        col_assignments = self.get_matched_points(cost_matrix)
        return None;

    def get_matched_points(self, cost_matrix):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        print(row_ind)
        print(col_ind)
        print(cost_matrix[row_ind, col_ind].sum())
        return col_ind

