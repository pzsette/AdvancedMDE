import pandas as pd

import utils
from MDE import MDE
from phases.Mutation import Mutator


def mde():
    points = pd.read_csv('points/points_reduced.csv')
    mde = MDE(points, k=5, population_size=10)
    solution = mde.execute_mdo()
    m = Mutator(solution, points)
    m.execute_mutation()
    utils.show_solution(points, solution)


if __name__ == '__main__':
    mde()
