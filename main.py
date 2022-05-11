import pandas as pd

import utils
from MDE import MDE


def mde():
    points = pd.read_csv('points/points_reduced.csv')
    mde_executor = MDE(points, n_clusters=5, population_size=10)
    solution = mde_executor.execute_mdo()
    utils.show_solution(solution)


if __name__ == '__main__':
    mde()
