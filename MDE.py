import random
import KMeans
from Population import Population
from phases.Crossover import Crossover
from phases.Mutation import Mutator


class MDE:
    def __init__(self,
                 points,
                 n_clusters,
                 population_size=5,
                 max_same_solution_repetition=1000,
                 min_population_diversity=5000,
                 do_mutation=False,
                 do_verbose=False
                 ):
        self.points = points
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.same_solution_repetition = 0
        self.max_same_solution_repetition = max_same_solution_repetition
        self.best_solution = None
        self.do_mutation = do_mutation
        self.min_population_diversity = min_population_diversity
        self.verboseprint = print if do_verbose else lambda *a, **k: None
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

                # Execute crossover
                # Select solutions for crossover
                crossover_solution = self.get_crossover_solution(index)
                # Get random solutions
                solution3 = p.get_solution(crossover_solution[0])
                solution2 = p.get_solution(crossover_solution[1])
                solution1 = p.get_solution(crossover_solution[2])
                # Execute Crossover
                offspring = Crossover.execute_crossover(points=self.points,
                                                        solution1=solution1,
                                                        solution2=solution2,
                                                        solution3=solution3)

                # Mutation
                if self.do_mutation:
                    m = Mutator(offspring, self.points)
                    offspring = m.execute_mutation()

                # Local optimization
                offspring = KMeans.compute_solution(self.points, self.n_clusters, start=offspring.coordinate_matrix)
                offspring.update_score()

                # Selection phase
                if offspring.get_score() < p.get_solution(index).get_score():
                    p.replace_solution(index, offspring)

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

    def get_crossover_solution(self, index):
        solutions = []
        while len(solutions) < 3:
            selected_index = random.randint(0, self.population_size - 1)
            if selected_index not in solutions and selected_index != index:
                solutions.append(selected_index)
        return solutions
