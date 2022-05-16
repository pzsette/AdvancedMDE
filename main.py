import pandas as pd

from MDE import MDE


def mde():
    points = pd.read_csv('points/liver.txt', sep=" ", header=None)

    mde_executor = MDE(points, n_clusters=5, population_size=150, max_same_solution_repetition=1000)
    solution = mde_executor.execute_mdo()
    print(f'Best score -> {solution.get_score()}')


if __name__ == '__main__':
    mde()
