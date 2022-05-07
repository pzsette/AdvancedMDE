import pandas as pd

import KMeans
import utils
from MDE import MDE


def mde():
    points = pd.read_csv('points/points_reduced.csv')
    k = 5
    mdo = MDE(points, k)
    solution = mdo.execute_mdo()
    #utils.show_solution(points, solution)
    #print(utils.sum_of_square(points, solution))


if __name__ == '__main__':
    mde()
