import unittest
import pandas as pd
import numpy as np

from models.Solution import Solution


class SolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._test_points = pd.read_csv('test_points/test_points.csv')
        membership_vector = np.array([0, 0, 1, 1])
        coordinate_matrix = np.array([[3.0, 0.0], [0.0, 3.0]])
        cls._s = Solution(membership_vector, coordinate_matrix)

    def test_get_score(self):
        self.assertEqual(self._s.get_score(self._test_points), 5.0)


