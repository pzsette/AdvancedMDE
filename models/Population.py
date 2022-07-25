import numpy as np
from local_opt import KMeans


class Population:
    def __init__(self, size, n_clusters, points):
        self.size = size
        self.points = points
        self.n_clusters = n_clusters
        self.solutions = [None]*size
        self.diff_population = np.zeros((size * (size - 1)) // 2)
        self.best_solution = None
        self.mean = 0

    def generate_solutions(self):
        # Init population with K-random execution of K-means
        for i in range(self.size):
            new_solution = KMeans.compute_solution(self.points, self.n_clusters)
            self.solutions[i] = new_solution
            self.replace_solution(index=i, new_solution=new_solution)
            self.mean += (new_solution.get_score() / self.n_clusters)

    def get_solution(self, index):
        return self.solutions[index]

    def replace_solution(self, index, new_solution):
        self.mean = self.mean - (self.get_solution(index).get_score() / self.n_clusters) + (new_solution.get_score() / self.n_clusters)
        self.solutions[index] = new_solution
        if self.best_solution is None or new_solution.get_score() < self.best_solution.get_score():
            self.best_solution = new_solution
        for i in range(index):
            if self.solutions[i] is not None:
                self.diff_population[i * self.size + index - (((i + 1) * (i + 2)) // 2)] = abs(self.solutions[i].get_score() - new_solution.get_score())
        for i in range(index + 1, self.size):
            if self.solutions[i] is not None:
                self.diff_population[index * self.size + i - (((index + 1) * (index + 2)) // 2)] = abs(
                    new_solution.get_score() - self.solutions[i].get_score())

    def get_best_solution(self):
        return self.best_solution

    def get_population_diversity(self):
        return sum(self.diff_population) / self.mean

    def print_population_scores(self):
        scores = []
        for solution in self.solutions:
            scores.append(solution.get_score())
        print(f'{self.get_population_diversity()} - {scores}')
