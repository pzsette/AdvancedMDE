import unittest
from phases.Crossover import *

from models.Solution import Solution


class CrossoverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        points = np.genfromtxt('test_points/test.txt')
        c_m1 = [[17664.525506, 3461.740121],
                [10727.393333, 4485.872745],
                [6656.675489, 3142.087880],
                [14871.742254, 6629.388685],
                [5379.450870, 6936.129752]]
        cls._solution1 = Solution(coordinate_matrix=np.asarray(c_m1), score=10, points=points)
        c_m2 = [[11627.755394, 4850.746805],
                [5348.346730, 6305.530616],
                [17643.981707, 3446.970691],
                [7848.105969, 3021.858010],
                [15348.106024, 7024.893434]]
        cls._solution2 = Solution(coordinate_matrix=np.asarray(c_m2), score=25, points=points)
        c_m3 = [[11627.755394, 4850.746805],
                [17643.981707, 3446.970691],
                [5348.346730, 6305.530616],
                [15348.106024, 7024.893434],
                [7848.105969, 3021.858010]]
        cls._solution3 = Solution(coordinate_matrix=np.asarray(c_m3), score=25, points=points)
        c_m4 = [[22, 19.9], [3.4, 4.5]]
        cls._solution4 = Solution(coordinate_matrix=np.asarray(c_m4), score=25, points=points)

    def test_solution_subtraction(self):
        sub = solutions_subtraction(self._solution3, self._solution4, 'exact')
        compo = np.asarray([[0.6, -0.5], [0.0, 2.1]], dtype=float)
        self.assertTrue((sub == compo).all())

    def test_execute_crossover(self):
        result = Crossover.execute_crossover(self._solution1, self._solution2, self._solution3, 'exact')
        print(result)
        self.assertTrue((result == np.array([15893.178286, 4855.165467], [7501.505570, 4779.264187])).all())
