import os

import pandas as pd

import KMeans
from MDE import MDE
import argparse


def mde(points, size, max_interation, n_clusters, do_mutation, do_verbose):
    mde_executor = MDE(
        points=points,
        n_clusters=n_clusters,
        population_size=size,
        max_same_solution_repetition=max_interation,
        do_mutation=do_mutation,
        do_verbose=do_verbose
    )
    solution = mde_executor.execute_mdo()
    print(f'  Best score -> {solution.get_score()}')
    print(f'  Number of K-MEANS executions -> {KMeans.compute_solution.calls}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Memetic Differential Evolution (MDE) for clustering")
    parser.add_argument("-d", type=str, help="dataset filename")
    parser.add_argument("-s", type=int, help="size of the population")
    parser.add_argument("-i", type=int, help="number of allowed consecutive iterations without improvements of the best solution")
    parser.add_argument("-c", type=int, help="number of clusters")
    parser.add_argument("-m", help="execute mutation phase", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")

    args = parser.parse_args()
    dataset = args.d
    population_size = args.s
    max_iteration = args.i
    clusters = args.c
    verbose = args.verbose
    mutation = args.m

    dataset_path = os.path.join('points', dataset)
    if not os.path.exists(dataset_path):
        raise ValueError(f'Can\'t find datset at the following path: {dataset_path}')
    points = pd.read_csv('points/' + dataset, sep=" ", header=None)

    if population_size < 4:
        raise argparse.ArgumentTypeError('Population size can\'t be less than 3')
    if clusters < 2:
        raise argparse.ArgumentTypeError('At least 2 clusters required')
    if clusters > len(points):
        raise argparse.ArgumentTypeError(f'The number of clusters must be less than {len(points)} (number of data points)')

    print(f'--Starting optimization with {dataset} | m = {clusters} clusters')
    mde(points=points,
        size=population_size,
        max_interation=max_iteration,
        n_clusters=clusters,
        do_mutation=mutation,
        do_verbose=verbose
        )


