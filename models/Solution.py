import sys

import utils
import numpy as np
from phases.Mutation import find_index


class Solution:
    def __init__(self, points, coordinate_matrix, score=None, membership_vector=None):
        self.coordinate_matrix = coordinate_matrix
        self.points = points
        self.membership_vector = membership_vector
        self._score = score

    def get_score(self):
        if self._score is None:
            raise Exception('Error evaluating score')
        return self._score

    # def compute_score(self):
    #     np.set_printoptions(threshold=sys.maxsize)
    #     n = len(self.points)
    #
    #     score = 0
    #     for i in range(n):
    #         assigned_centroid = self.coordinate_matrix[self.get_membership_vector()[i]]
    #         dst = utils.euclidean_distance(self.points[i], assigned_centroid)
    #         square = np.square(dst)
    #         score += square
    #     return score

    def solution_repair(self, m):
        n = len(self.points)
        sizes = [0] * m
        empty_clusters = []

        for assignment in self.get_membership_vector():
            sizes[assignment] += 1

        for i in range(m):
            if sizes[i] == 0:
                empty_clusters.append(i)

        if len(empty_clusters) > 0:
            dist_centroid = []
            total_dist = 0

            self.assignment_to_centroid()

            for i in range(n):
                dist_centroid.append(utils.euclidean_distance(self.points[i], self.coordinate_matrix[self.membership_vector[i]]))
                total_dist = total_dist + dist_centroid[i]

            pr = [utils.pr(dist_centroid[0], total_dist, 0.5, n)]
            for i in range(n):
                pr.append(pr[i-1] + utils.pr(dist_centroid[i], total_dist, 0.5, n))
            e = 0
            while e < len(empty_clusters):
                r = np.random.uniform(0.0, pr[-1])
                p = find_index(pr, r, 0, n-1) + 1
                if sizes[self.membership_vector[p]] > 1:
                    sizes[self.membership_vector[p]] -= 1
                    self.membership_vector[p] = empty_clusters[e]
                    e += 1
            self.assignment_to_centroid()
        else:
            self.assignment_to_centroid()

    def get_membership_vector(self):
        if self.membership_vector is None:
            self.membership_vector = utils.get_memb_vect_from_coord_matrix(self.points, np.array(self.coordinate_matrix))
        return self.membership_vector

    def assignment_to_centroid(self):
        n = len(self.points)
        m = len(self.coordinate_matrix)
        d = len(self.points[0])
        self.coordinate_matrix = np.zeros((m, d))
        sizes = [0] * m

        for i in range(n):
            sizes[self.membership_vector[i]] += 1
            for j in range(d):
                self.coordinate_matrix[self.membership_vector[i]][j] = self.coordinate_matrix[self.membership_vector[i]][j] + self.points[i][j]

        for i in range(m):
            for j in range(d):
                if sizes[i] != 0:
                    self.coordinate_matrix[i][j] = (self.coordinate_matrix[i][j]) / sizes[i]

    def centroid_to_assignment(self):
        n = len(self.points)
        m = len(self.coordinate_matrix)

        m_vect = []

        for i in range(n):
            min_dst = sys.float_info.max
            index = -1
            for j in range(m):
                dst = utils.euclidean_distance(self.points[i], self.coordinate_matrix[j])
                if dst < min_dst:
                    min_dst = dst
                    index = j
            if index == -1 or index >= m:
                raise ValueError("Error index value")
            m_vect.append(index)
        return m_vect

    def remove_center(self, index):
        list_points = []

        for assignment_index, assignment in enumerate(self.get_membership_vector()):
            if assignment == index:
                list_points.append(assignment_index)

        # print(list_points)

        # print('lista')
        # print(list_points)
        # print(self.membership_vector)

        for assignment_index in list_points:
            min_dst = sys.float_info.max
            for index_centroid in range(len(self.coordinate_matrix)):
                if index_centroid != index:
                    dst = utils.euclidean_distance(self.points[assignment_index], self.coordinate_matrix[index_centroid])
                    if dst < min_dst:
                        min_dst = dst
                        self.membership_vector[assignment_index] = index_centroid

        # print(self.membership_vector)

    def reinsert_center(self, index_to_replace, point_index):

        new_point = self.points[point_index]
        for i in range(len(self.coordinate_matrix[index_to_replace])):
            self.coordinate_matrix[index_to_replace][i] = new_point[i]
        # print(self.coordinate_matrix)

        for i in range(len(self.points)):
            dst = utils.euclidean_distance(new_point, self.points[i])
            if dst < utils.euclidean_distance(self.points[i], self.coordinate_matrix[self.membership_vector[i]]):
                self.membership_vector[i] = index_to_replace

    def print_coor_with_curly(self):
        print('{')
        for centroid in self.coordinate_matrix:
            centroide = '{'
            for index, point in enumerate(centroid):
                centroide += str(point)
                if index != (len(centroid) - 1):
                    centroide += ', '
            centroide += '}, '
            print(centroide)
        print('}')

    def da_repair(self, m):
        sizes = [0] * m
        empty_clusters = []
        for assignment in self.get_membership_vector():
            sizes[assignment] += 1

        for i in range(m):
            if sizes[i] == 0:
                empty_clusters.append(i)
        if len(empty_clusters) > 0:
            print("REPPPP")








