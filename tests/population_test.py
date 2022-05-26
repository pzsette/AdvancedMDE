import unittest
import pandas as pd
import numpy as np

from models.Solution import Solution
from models.Population import Population


class PopulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._test_points = pd.read_csv('test_points/test_points.csv')

        membership_vector = np.array([0, 0, 1, 1])
        coordinate_matrix = np.array([[3.0, 0.0], [0.0, 3.0]])
        cls._s1 = Solution(membership_vector, coordinate_matrix)

        membership_vector = np.array([0, 0, 1, 1])
        coordinate_matrix = np.array([[3.0, -3.0], [-2.0, 3.0]])
        cls._s2 = Solution(membership_vector, coordinate_matrix)

        cls._p = Population(2, 2, cls._test_points)
        cls._p.generate_solutions()
        cls._p.replace_solution(0, cls._s1)
        cls._p.replace_solution(1, cls._s2)

    def test_get_solution(self):
        s = self._p.get_solution(1)
        self.assertTrue((s.membership_vector == np.array([0, 0, 1, 1])).all())
        self.assertTrue((s.coordinate_matrix == np.array([[3.0, -3.0], [-2.0, 3.0]])).all())

    def test_get_best_solution(self):
        s = self._p.get_best_solution()
        self.assertEqual(s.get_score(self._test_points), 5.0)
        self.assertTrue((s.membership_vector == np.array([0, 0, 1, 1])).all())
        self.assertTrue((s.coordinate_matrix == np.array([[3.0, 0.0], [0.0, 3.0]])).all())
