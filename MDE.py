import random

import numpy as np
from scipy.optimize import linear_sum_assignment

import KMeans
import utils
from Population import Population
from Solution import Solution


class MDE:
    def __init__(self, points, n_clusters, population_size=10, f=0.5, max_iteration=3, max_same_solution_repetition=2):
        self.points = points
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.f = f
        self.iteration = 0
        self.same_solution_repetition = 0
        self.max_iteration = max_iteration
        self.max_same_solution_repetition = max_same_solution_repetition
        self.best_solution = None

    def execute_mdo(self):
        p = Population(size=self.population_size, n_clusters=self.n_clusters, points=self.points)
        p.generate_solutions()

        # Loop until stopping one stopping criterion is not satisfied
        while self.check_stopping_criterion(p):
            # Crossover
            for index, solution in enumerate(p.solutions):
                print(f'Solution -> {index}')
                print(f'Executing crossover...')
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
                for point in sub:
                    point[0] = self.f * point[0]
                    point[1] = self.f * point[1]
                # Sum S1 + F(S2 - S3)
                cost_matrix = utils.build_bipartite_graph(sub, solution3.coordinate_matrix)
                matched_points = self.get_matched_points(cost_matrix)
                offspring_coordinate_matrix = np.empty(shape=(self.n_clusters, 2))
                for index_centroid in range(self.n_clusters):
                    points_sum = utils.sum_points(solution1.coordinate_matrix[index_centroid],
                                                  sub[matched_points[index_centroid]])
                    np.append(offspring_coordinate_matrix, points_sum)
                # Build solution
                offspring_membership_vector = self.get_memb_vect_from_coord_matrix(offspring_coordinate_matrix)
                offspring = Solution(offspring_membership_vector, offspring_coordinate_matrix)

                # Local optimization
                #print('Executing local optimization...')
                l_offspring = KMeans.compute_solution(self.points, self.n_clusters, start=offspring.coordinate_matrix)

                # Mutation
                #print('Executing mutation...')

                # Selection phase
                print('Executing selection...')
                if l_offspring.get_score(self.points) < p.get_solution(index).get_score(self.points):
                    p.replace_solution(index, offspring)
                    print(f'Solution {index} replaced with offspring')
                print('------------------------------------')

            print('Computing best solution among population...')
            solution = p.get_best_solution()
            if self.best_solution is None:
                print('First solution!')
                self.best_solution = solution
            elif solution.get_score(self.points) == self.best_solution.get_score(self.points):
                print('Repetition!')
                self.same_solution_repetition = self.same_solution_repetition + 1
            else:
                print(f'Best solution improved {self.best_solution.get_score(self.points)} '
                      f'-> {solution.get_score(self.points)}')
                self.best_solution = solution
                self.same_solution_repetition = 0
            print(f'Best score -> {self.best_solution.get_score(self.points)}')
        return self.best_solution

    def check_stopping_criterion(self, p):
        # Population diversity falls below a threshold
        if p.get_population_diversity() < 50000:
            print('Terminated due to low population diversity!')
            return False
        if self.same_solution_repetition >= self.max_same_solution_repetition:
            print(f'Terminate due to {self.max_same_solution_repetition} best solution repetitions')
            return False
        # Max consecutive iterations performed without any improvement in the best solution
        if self.iteration >= self.max_iteration:
            return False

        print(f'Iteration: {self.iteration}')
        self.iteration = self.iteration + 1
        return True

    def get_crossover_solution(self, index):
        solutions = []
        while len(solutions) < 3:
            selected_index = random.randint(0, self.population_size-1)
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

    def get_memb_vect_from_coord_matrix(self, coordinate_matrix):
        membership_vector = []
        for row in self.points.iterrows():
            assigned_centroid = 0
            x_point = float(row[1][0])
            y_point = float(row[1][1])
            min_dst = utils.euclidean_distance(x_point, y_point, coordinate_matrix[0][0], coordinate_matrix[0][1])
            for index, centroid in enumerate(coordinate_matrix):
                dst = utils.euclidean_distance(x_point, y_point, centroid[0], centroid[1])
                if dst < min_dst:
                    assigned_centroid = index
                    min_dst = dst
            membership_vector.append(assigned_centroid)
        return np.asarray(membership_vector)




