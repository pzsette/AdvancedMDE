from local_opt import KMeans
import utils
from models.Population import Population
from phases import DSelection
from phases.GreedyGeneration import greedy_generation


class DMDE:
    def __init__(self,
                 points,
                 n_clusters,
                 population_size=5,
                 max_same_solution_repetition=1000,
                 min_population_diversity=5000,
                 do_verbose=True,
                 ):
        self.points = points
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.same_solution_repetition = 0
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
        while self.check_stopping_criterion(p):
            for index, solution in enumerate(p.solutions):

                # Greedy generation
                index_to_compare = utils.get_random_in_range_less_one(self.population_size, index)
                solution = p.get_solution(index)
                solution_to_compare = p.get_solution(index_to_compare)
                generated_coordinate_matrix = greedy_generation(solution, solution_to_compare, 0.5)

                # Local optimization
                candidate_solution = KMeans.compute_solution(self.points, self.n_clusters, start=generated_coordinate_matrix)

                DSelection.dmde_selection(p, candidate_solution)

            solution = p.get_best_solution()
            if solution.get_score() == self.best_solution.get_score():
                self.verboseprint('  Repetition!')
                self.same_solution_repetition = self.same_solution_repetition + 1
            else:
                self.verboseprint(f'  Best solution improved {self.best_solution.get_score()} -> {solution.get_score()}')
                self.best_solution = solution
                self.same_solution_repetition = 0

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
