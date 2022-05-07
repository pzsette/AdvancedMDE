import KMeans


class MDO:
    def __init__(self, points, n_clusters):
        self.points = points
        self.n_clusters = n_clusters

    def execute_mdo(self):
        solution = KMeans.compute_solution(self.points, self.n_clusters)
        return solution
