from sklearn.cluster import KMeans

from Solution import Solution


def compute_solution(points, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='random')
    kmeans.fit(points)
    kmeans.fit_predict(points)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)
    solution = Solution(kmeans.labels_, kmeans.cluster_centers_)
    return solution
