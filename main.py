import pandas as pd

import utils
from MDO import MDO


def mde():
    points = pd.read_csv('points/points_reduced.csv')
    mdo = MDO(points, 5)
    solution = mdo.execute_mdo()
    utils.show_solution(points, solution)


if __name__ == '__main__':
    mde()
