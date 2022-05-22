import unittest

from matching import GreedyMatching


class CrossoverTest(unittest.TestCase):
    def test_greedy_matching(self):
        assigned = GreedyMatching.greedy_matching([[0, 0], [3, 3], [20, 20]], [[4, 4], [21, 21], [0.1, 0.1]])
        self.assertEqual(assigned, [2, 0, 1])
