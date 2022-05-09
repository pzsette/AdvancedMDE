import KMeans


class Population:
    def __init__(self, size, n_clusters, points):
        self.size = size
        self.points = points
        self.n_clusters = n_clusters
        self.solutions = []

    def generate_solutions(self):
        # Init population with K-random execution of K-means
        for i in range(self.size):
            self.solutions.append(KMeans.compute_solution(self.points, self.n_clusters))

    def get_solution(self, index):
        return self.solutions[index]

    def replace_solution(self, index, new_solution):
        self.solutions[index] = new_solution

    def get_best_solution(self):
        scores = []
        best = self.solutions[0]
        scores.append(best.get_score(self.points))
        for solution in (self.solutions[1:]):
            scores.append(solution.get_score(self.points))
            if solution.get_score(self.points) < best.get_score(self.points):
                best = solution
        print(scores)
        return best



