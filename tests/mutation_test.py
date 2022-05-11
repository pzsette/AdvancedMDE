import unittest

import numpy as np

from phases.Mutation import get_close_centroid_index


class MutatorTest(unittest.TestCase):
    def test_get_close_centroid_index(self):
        coord_matrix = np.asarray([[1, 1], [2, 2], [3, 3]])
        self.assertEqual(0, get_close_centroid_index(0, 0, coord_matrix))
        self.assertEqual(1, get_close_centroid_index(2, 2, coord_matrix))
        self.assertEqual(2, get_close_centroid_index(3, 3, coord_matrix))
        self.assertEqual(2, get_close_centroid_index(6, 6, coord_matrix))
