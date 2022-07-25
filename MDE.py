import time

import utils
from local_opt import KMeans
from models.Population import Population
from models.Solution import Solution
from phases.Crossover import Crossover
from phases.Mutation import Mutator
import numpy as np


class MDE:
    def __init__(self,
                 points,
                 n_clusters,
                 population_size=5,
                 max_same_solution_repetition=1000,
                 min_population_diversity=0.00001,
                 do_mutation=False,
                 matching_type='exact',
                 do_verbose=False):
        self.points = points
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.same_solution_repetition = 0
        self.max_same_solution_repetition = max_same_solution_repetition
        self.best_solution = None
        self.do_mutation = do_mutation
        self.matching_type = matching_type
        self.min_population_diversity = min_population_diversity
        self.verboseprint = print if do_verbose else lambda *a, **k: None
        if self.population_size is not None and self.population_size < 4:
            raise Exception(f'Population size must be 4 or higher')

    def execute_mdo(self):
        print(f'--Starting MDE optimization | m = {self.n_clusters} clusters')

        p = Population(size=self.population_size, n_clusters=self.n_clusters, points=self.points)
        p.generate_solutions()
        p.print_population_scores()
        self.best_solution = p.get_best_solution()
        # Loop until stopping one stopping criterion is not satisfied
        counter = 0
        first_time = True
        while first_time is True or self.check_stopping_criterion(p):
            first_time = False
            counter += 1
            for index, solution in enumerate(p.solutions):
                crossover_solution_indexes = self.get_crossover_solution(index)
                # Get random solutions
                j = crossover_solution_indexes[0]
                k = crossover_solution_indexes[1]
                z = crossover_solution_indexes[2]
                # Execute Crossover
                c = Crossover(p, index1=j, index2=k, index3=z, matching_type=self.matching_type)
                offspring_coord_matrix = c.execute_crossover()

                offspring_solution = Solution(points=self.points, coordinate_matrix=offspring_coord_matrix)

                # Mutation
                if self.do_mutation:
                    m = Mutator(offspring_solution)
                    m.execute_mutation()

                # Repair
                offspring_solution.solution_repair(self.n_clusters)

                # Local optimization
                offspring_solution = KMeans.compute_solution(self.points, self.n_clusters, start=offspring_solution.coordinate_matrix)

                # Selection phase
                if offspring_solution.get_score() < p.get_solution(index).get_score():
                    p.replace_solution(index, offspring_solution)
                    if offspring_solution.get_score() < self.best_solution.get_score():
                        self.verboseprint(f'  Best solution improved {self.best_solution.get_score()} -> {offspring_solution.get_score()}')
                        # p.print_population_scores()
                        self.best_solution = offspring_solution
                        self.same_solution_repetition = 0
                    else:
                        self.same_solution_repetition = self.same_solution_repetition + 1
                else:
                    self.same_solution_repetition = self.same_solution_repetition + 1

        return self.best_solution

    def check_stopping_criterion(self, p):
        # Population diversity falls below a threshold
        population_diversity = p.get_population_diversity()
        if population_diversity < self.min_population_diversity:
            print('--Terminated due to low population diversity!')
            p.print_population_scores()
            return False
        # Max consecutive iterations performed without any improvement in the best solution
        if self.same_solution_repetition >= self.max_same_solution_repetition:
            print(f'--Terminate due to {self.max_same_solution_repetition} best solution repetitions')
            return False
        return True

    def get_crossover_solution(self, index):
        solutions = []
        while len(solutions) < 3:
            selected_index = np.random.randint(0, self.population_size)
            if selected_index not in solutions and selected_index != index:
                solutions.append(selected_index)
        return solutions
