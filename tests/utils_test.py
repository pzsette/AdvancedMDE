import unittest

import numpy as np

import utils


class UtiliTest(unittest.TestCase):

    def test_euclidean_distance(self):
        self.assertEqual(utils.euclidean_distance(0.0, 0.0, 4.0, 0.0), 4.0)

    def test_euclidean_distance_with_coincident_points(self):
        self.assertEqual(utils.euclidean_distance(0.0, 0.0, 0.0, 0.0), 0.0)

    def test_trivial_build_bipartite_graph(self):
        b_graph = utils.build_bipartite_graph(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))
        self.assertEqual(b_graph, 1.0)

    def test_build_bipartite_graph(self):
        b_graph = utils.build_bipartite_graph(np.array([[1.0, 1.0], [-1.0, -1.0]]), np.array([[-1.0, 1.0], [1.0, -1.0]]))
        print(b_graph)
        self.assertTrue((b_graph == np.array([[2.0, 2.0], [2.0, 2.0]])).all())
