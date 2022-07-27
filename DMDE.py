from local_opt import KMeans
import utils
from models.Population import Population
from models.Solution import Solution
from phases import DSelection
from phases.GreedyGeneration import greedy_generation
import numpy as np


class DMDE:
    def __init__(self,
                 points,
                 n_clusters,
                 population_size=5,
                 max_same_solution_repetition=1000,
                 matching_type='exact',
                 min_population_diversity=0.00001,
                 do_verbose=True,
                 ):
        self.points = points
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.same_solution_repetition = 0
        self.matching_type = matching_type
        self.max_same_solution_repetition = max_same_solution_repetition
        self.min_population_diversity = min_population_diversity
        self.best_solution = None,
        self.verboseprint = print if do_verbose else lambda *a, **k: None

    def execute_d_mde(self):
        print(f'--Starting D-MDE optimization | m = {self.n_clusters} clusters')
        p = Population(size=self.population_size, n_clusters=self.n_clusters, points=self.points)
        p.generate_solutions()
        self.best_solution = p.get_best_solution()
        # Loop until stopping one stopping criterion is not satisfied
        first_time = True
        while first_time is True or self.check_stopping_criterion(p):
            first_time = False
            for index, solution in enumerate(p.solutions):

                # Greedy generation
                index_to_compare = utils.get_random_in_range_less_one(self.population_size, index)
                solution = p.get_solution(index)
                solution_to_compare = p.get_solution(index_to_compare)
                _, generated_coordinate_matrix = greedy_generation(solution=solution,
                                                                   solution_to_compare=solution_to_compare,
                                                                   f=np.random.uniform(low=0.5, high=0.8),
                                                                   matching_type=self.matching_type)

                # Repair
                offspring_solution = Solution(points=self.points, coordinate_matrix=generated_coordinate_matrix)
                offspring_solution.solution_repair(self.n_clusters)

                # Local optimization
                candidate_solution = KMeans.compute_solution(self.points,
                                                             self.n_clusters,
                                                             start=offspring_solution.coordinate_matrix)

                index_to_compare = DSelection.dmde_selection(p, candidate_solution)

                # Selection phase
                if candidate_solution.get_score() < p.get_solution(index_to_compare).get_score():
                    p.replace_solution(index_to_compare, candidate_solution)
                    if candidate_solution.get_score() < self.best_solution.get_score():
                        self.verboseprint(
                            f'  Best solution improved {self.best_solution.get_score()} -> {candidate_solution.get_score()}')
                        self.best_solution = candidate_solution
                        self.same_solution_repetition = 0
                    else:
                        self.same_solution_repetition = self.same_solution_repetition + 1
                else:
                    self.same_solution_repetition = self.same_solution_repetition + 1

        return self.best_solution

    def check_stopping_criterion(self, p):
        # Population diversity falls below a threshold
        if p.get_population_diversity() < self.min_population_diversity:
            print('--Terminated due to low population diversity!')
            return False
        # Max consecutive iterations performed without any improvement in the best solution
        if self.same_solution_repetition >= self.max_same_solution_repetition:
            print(f'--Terminate due to {self.max_same_solution_repetition} best solution repetitions')
            return False
        return True
