import random

import numpy as np
from scipy.optimize import linear_sum_assignment

import KMeans
import utils
from Solution import Solution


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

        #for index, solution in enumerate(population):
        #    print(f'index {index}')
        #    print(self.get_crossover_solution(index))

        # Loop until stopping one stopping criterion is not satisfied
        #while self.check_stopping_criterion():

        # Crossover
        for index, solution in enumerate(population):
            # Select solutions for crossover
            crossover_solution = self.get_crossover_solution(index)

            # Execute crossover

            # get random solutions
            solution3 = population[crossover_solution[0]]
            solution2 = population[crossover_solution[1]]
            solution1 = population[crossover_solution[2]]
            # Subtraction (S2 - S3)
            sub = self.solutions_subtraction(solution2, solution3)
            # Function f
            for point in sub:
                point[0] = self.f * point[0]
                point[1] = self.f * point[1]
            # Sum S1 + F(S2 - S3)
            cost_matrix = utils.build_bipartite_graph(sub, solution3.coordinate_matrix)
            matched_points = self.get_matched_points(cost_matrix)
            offspring_coordinate_matrix = np.empty(shape=(self.n_clusters, 2))
            for index_centroid in range(self.n_clusters):
                x = solution1.coordinate_matrix[index_centroid][0] + \
                    sub[matched_points[index_centroid]][0]
                y = solution1.coordinate_matrix[index_centroid][1] + \
                    sub[matched_points[index_centroid]][1]
                np.append(offspring_coordinate_matrix, [x, y])

            # Build solution
            offspring_membership_vector = self.get_memb_vect_from_coord_matrix(offspring_coordinate_matrix)
            offspring = Solution(offspring_membership_vector, offspring_coordinate_matrix)

            # Mutation

            if offspring.get_score(self.points) < population[index].get_score(self.points):
                population[index] = offspring

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
        cost_matrix = utils.build_bipartite_graph(solution1.coordinate_matrix, solution2.coordinate_matrix)
        col_assignments = self.get_matched_points(cost_matrix)
        new_coordinate_matrix = np.empty(shape=(self.n_clusters, 2))
        for index_centroid in range(self.n_clusters):
            x = solution1.coordinate_matrix[index_centroid][0] - \
                solution2.coordinate_matrix[col_assignments[index_centroid]][0]
            y = solution1.coordinate_matrix[index_centroid][1] - \
                solution2.coordinate_matrix[col_assignments[index_centroid]][1]
            np.append(new_coordinate_matrix, [x, y])
        return new_coordinate_matrix

    def get_matched_points(self, cost_matrix):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        #print(row_ind)
        #print(col_ind)
        #print(cost_matrix[row_ind, col_ind].sum())
        return col_ind

    def get_memb_vect_from_coord_matrix(self, coordinate_matrix):
        print(coordinate_matrix.shape)

        membership_vector = []
        for row in self.points.iterrows():
            assigned_centroid = 0
            x_point = float(row[1][0])
            y_point = float(row[1][1])
            min_dst = utils.euclidean_distance(x_point, y_point, coordinate_matrix[0][0], coordinate_matrix[0][1])
            for index, centroid in enumerate(coordinate_matrix):
                dst = utils.euclidean_distance(x_point, y_point, centroid[0], centroid[1])
                #print("check con n" + str(index) + " " + str(centroid[0]) + "," + str(centroid[1])+" dst: "+str(dst))
                if dst < min_dst:
                    assigned_centroid = index
                    min_dst = dst
            membership_vector.append(assigned_centroid)
        return np.asarray(membership_vector)




