import numpy as np

from local_opt import KMeans
import time


class Population:
    def __init__(self, size, n_clusters, points):
        self.size = size
        self.points = points
        self.n_clusters = n_clusters
        self.solutions = [None]*size
        self.diff_population = np.zeros((size * (size - 1)) // 2)

    def generate_solutions(self):
        # Init population with K-random execution of K-means
        for i in range(self.size):
            new_solution = KMeans.compute_solution(self.points, self.n_clusters)
            self.solutions[i] = new_solution
            self.replace_solution(index=i, new_solution=new_solution)

    def get_solution(self, index):
        return self.solutions[index]

    def replace_solution(self, index, new_solution):
        self.solutions[index] = new_solution
        for i in range(index):
            if self.solutions[i] is not None:
                valle = i * self.size + index - (((i + 1) * (i + 2)) / 2)
                self.diff_population[int(valle)] = abs(self.solutions[i].get_score() - new_solution.get_score())
        for i in range(index + 1, self.size):
            if self.solutions[i] is not None:
                self.diff_population[index * self.size + i - (((index + 1) * (index + 2)) // 2)] = abs(
                    new_solution.get_score() - self.solutions[i].get_score())

    def get_best_solution(self):
        scores = []
        best = self.solutions[0]
        scores.append(best.get_score())
        for solution in (self.solutions[1:]):
            scores.append(solution.get_score())
            if solution.get_score() < best.get_score():
                best = solution
        return best

    def get_population_diversity(self):
        return sum(self.diff_population)

    def print_population_scores(self):
        scores = []
        for solution in self.solutions:
            scores.append(solution.get_score())
        print(f'sum: {sum(scores)} - {scores}')
