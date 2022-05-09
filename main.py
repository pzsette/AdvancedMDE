import pandas as pd

import utils
from MDE import MDE


def mde():
    points = pd.read_csv('points/points_reduced.csv')
    k = 5
    mde = MDE(points, k)
    solution = mde.execute_mdo()
    utils.show_solution(points, solution)


if __name__ == '__main__':
    mde()
