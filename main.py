import os
import random

import numpy as np
import pandas as pd

import utils
from local_opt import KMeans
from DMDE import DMDE
from MDE import MDE
from GMDE import GMDE
import argparse


def mde(points, size, max_interation, n_clusters, do_mutation, matching_type, do_verbose):
    mde_executor = MDE(
        points=points,
        n_clusters=n_clusters,
        population_size=size,
        max_same_solution_repetition=max_interation,
        do_mutation=do_mutation,
        matching_type=matching_type,
        do_verbose=do_verbose
    )
    solution = mde_executor.execute_mdo()
    utils.print_result(solution.get_score(), KMeans.compute_solution.calls)


def gmde(points, size, max_iteration, n_clusters, do_verbose):
    gmde_executor = GMDE(
        points=points,
        n_clusters=n_clusters,
        population_size=size,
        max_same_solution_repetition=max_iteration,
        do_verbose=do_verbose
    )
    solution = gmde_executor.execute_g_mde()
    utils.print_result(solution.get_score(), KMeans.compute_solution.calls)


def dmde(points, size, max_iteration, n_clusters, do_verbose):
    dmde_executor = DMDE(
        points=points,
        n_clusters=n_clusters,
        population_size=size,
        max_same_solution_repetition=max_iteration,
        do_verbose=do_verbose
    )
    solution = dmde_executor.execute_d_mde()
    utils.print_result(solution.get_score(), KMeans.compute_solution.calls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Memetic Differential Evolution (MDE) for clustering")
    parser.add_argument("-d", type=str, help="dataset filename")
    parser.add_argument("-s", type=int, help="size of the population")
    parser.add_argument("-i", type=int, help="number of allowed consecutive iterations without improvements of the best solution")
    parser.add_argument("-c", type=int, help="number of clusters")
    parser.add_argument('-matching',
                        default='exact',
                        choices=['exact', 'greedy'],
                        help='matching algorithm')
    parser.add_argument("-m", help="execute mutation", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")

    args = parser.parse_args()
    dataset = args.d
    population_size = args.s
    max_iteration = args.i
    clusters = args.c
    verbose = args.verbose
    mutation = args.m
    matching = args.matching

    dataset_path = os.path.join('points', dataset)
    if not os.path.exists(dataset_path):
        raise ValueError(f'Can\'t find dataset at the following path: {dataset_path}')
    points = pd.read_csv('points/' + dataset, sep=" ", header=None)

    if population_size < 4:
        raise argparse.ArgumentTypeError('Population size can\'t be less than 3')
    if clusters < 2:
        raise argparse.ArgumentTypeError('At least 2 clusters required')
    if clusters > len(points):
        raise argparse.ArgumentTypeError(f'The number of clusters must be less than {len(points)} (number of data points)')

    #random.seed(1234)
    #np.random.seed(1234)

    #print(random.random())

    mde(points=points,
        size=population_size,
        max_interation=max_iteration,
        n_clusters=clusters,
        do_mutation=mutation,
        matching_type=matching,
        do_verbose=verbose
        )

    #gmde(points=points, size=population_size, max_iteration=max_iteration, n_clusters=clusters, do_verbose=verbose)

    #dmde(points=points, size=population_size, max_iteration=max_iteration, n_clusters=clusters, do_verbose=verbose)


