import unittest

import numpy as np

from models.Solution import Solution
from phases import GreedyGeneration
import pandas as pd


class GenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        points = pd.read_csv('test_points/test.txt', sep=" ", header=None)
        c_m1 = [[1, 1], [20, 20]]
        cls._solution1 = Solution(coordinate_matrix=c_m1, score=10, points=points)
        c_m2 = [[3, 3], [21, 21]]
        cls._solution2 = Solution(coordinate_matrix=c_m2, score=25, points=points)

    def test_greedy_matching(self):
        phi, assigned = GreedyGeneration.greedy_generation(self._solution1, self._solution2, 0.5)
        phi2, assigned2 = GreedyGeneration.greedy_generation(self._solution2, self._solution1, 0.5)
        self.assertEqual(phi, -1)
        self.assertEqual(phi2, 1)
        self.assertTrue((assigned == np.array([[0, 0], [19.5, 19.5]])).all())
        print(assigned2)
        self.assertTrue((assigned2 == np.array([[2, 2], [20.5, 20.5]])).all())
