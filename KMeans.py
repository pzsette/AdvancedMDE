from sklearn.cluster import KMeans

from Solution import Solution


def compute_solution(points, n_clusters, start=None):
    if start is None:
        kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init=start, n_init=1)
    kmeans.fit(points)
    kmeans.fit_predict(points)
    solution = Solution(points=points, coordinate_matrix=kmeans.cluster_centers_, membership_vector=kmeans.labels_)
    return solution
