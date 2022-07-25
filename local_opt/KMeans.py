import numpy as np
from sklearn.cluster import KMeans
from models.Solution import Solution


def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


@counted
def compute_solution(points, n_clusters, start=None):
    if start is None:
        kseed = np.random.randint(50000)
        kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1, random_state=kseed)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init=start, n_init=1)
    kmeans.fit(points)
    solution = Solution(points=points, coordinate_matrix=kmeans.cluster_centers_, score=kmeans.inertia_, membership_vector=kmeans.labels_)
    return solution


def get_score():
    count = compute_solution.calls
    compute_solution.calls = 0
    return count

