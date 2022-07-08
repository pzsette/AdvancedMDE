import unittest
import numpy as np

import utils


class UtilsTest(unittest.TestCase):

    def test_euclidean_distance(self):
        self.assertEqual(utils.euclidean_distance(0.0, 0.0, 4.0, 0.0), 4.0)

    def test_euclidean_distance_with_coincident_points(self):
        self.assertEqual(utils.euclidean_distance(0.0, 0.0, 0.0, 0.0), 0.0)

    def test_trivial_build_bipartite_graph(self):
        b_graph = utils.build_bipartite_graph(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))
        self.assertEqual(b_graph, 1.0)

    def test_build_bipartite_graph(self):
        b_graph = utils.build_bipartite_graph(np.array([[1.0, 1.0], [-1.0, -1.0]]), np.array([[-1.0, 1.0], [1.0, -1.0]]))
        self.assertTrue((b_graph == np.array([[2.0, 2.0], [2.0, 2.0]])).all())

    def test_subtract_points(self):
        self.assertNotEqual(utils.subtract_points([3.5, 3.5], [1.2, 1.2]), [4.7, 4.7])
        self.assertEqual(utils.subtract_points([3.5, 3.5], [1.2, 1.2]), [2.3, 2.3])

    def test_sum_points(self):
        self.assertNotEqual(utils.sum_points([3.5, 3.5], [1.2, 1.2]), [2.3, 2.3])
        self.assertEqual(utils.sum_points([3.5, 3.5], [1.2, 1.2]), [4.7, 4.7])

    def test_get_memb_vect_from_coord_matrix(self):
        memb_vect = utils.get_memb_vect_from_coord_matrix(np.genfromtxt('test_points/test.txt'), np.array([[1.1, 1.1], [3.5, 3.5]]))
        self.assertTrue((memb_vect == [0, 1, 1]).all())
