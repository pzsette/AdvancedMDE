import os
import time
import numpy as np

import utils
from HMDE import HMDE
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
    return mde_executor.execute_mdo()


def gmde(points, size, max_iteration, n_clusters, matching_type, do_verbose):
    gmde_executor = GMDE(
        points=points,
        n_clusters=n_clusters,
        population_size=size,
        max_same_solution_repetition=max_iteration,
        matching_type=matching_type,
        do_verbose=do_verbose
    )
    return gmde_executor.execute_g_mde()


def dmde(points, size, max_iteration, n_clusters, matching_type, do_verbose):
    dmde_executor = DMDE(
        points=points,
        n_clusters=n_clusters,
        population_size=size,
        max_same_solution_repetition=max_iteration,
        matching_type=matching_type,
        do_verbose=do_verbose
    )
    return dmde_executor.execute_d_mde()


def hmde(points, size, max_iteration, n_clusters, matching_type, do_verbose):
    hmde_executor = HMDE(
        points=points,
        n_clusters=n_clusters,
        population_size=size,
        max_same_solution_repetition=max_iteration,
        matching_type=matching_type,
        do_verbose=do_verbose
    )
    return hmde_executor.execute_h_mde()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Memetic Differential Evolution (MDE) for clustering")
    parser.add_argument('-method',
                        default='MDE',
                        choices=['MDE', 'GMDE', 'DMDE', 'HMDE'],
                        help='MDE variant')
    parser.add_argument("-s", type=int, help="size of the population")
    parser.add_argument("-i", type=int, help="number of allowed consecutive iterations without improvements of the"
                                             " best solution")

    parser.add_argument('-matching',
                        default='exact',
                        choices=['exact', 'greedy'],
                        help='matching algorithm')
    parser.add_argument("-m", help="execute mutation", action="store_true")
    parser.add_argument("--d", type=str, help="dataset filename")
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")

    args = parser.parse_args()
    method = args.method
    if args.d is not None:
        datasets = [args.d]
    else:
        datasets = ['ionosphere.txt', 'page.txt', 'pendigit.txt']
    population_size = args.s
    max_iteration = args.i
    verbose = args.verbose
    mutation = args.m
    matching = args.matching

    for dataset in datasets:
        dataset_path = os.path.join('points', dataset)
        if not os.path.exists(dataset_path):
            raise ValueError(f'Can\'t find dataset at the following path: {dataset_path}')
        print('#########')
        print('DATASET:', dataset)
        print('#########')
        points = np.genfromtxt(dataset_path)

        if population_size < 4:
            raise argparse.ArgumentTypeError('Population size can\'t be less than 3')

        for seed in [16007, 10000, 12345, 00000, 56789]:
            np.random.seed(seed)
            print('#########')
            print('SEED:', seed)
            print('#########')

            for cluster_seq in [2, 5, 10, 15, 30]:
                print('CLUSTER:', cluster_seq)

                start_time = time.time()

                if method == 'MDE':
                    solution = mde(points=points,
                                   size=population_size,
                                   max_interation=max_iteration,
                                   n_clusters=cluster_seq,
                                   do_mutation=mutation,
                                   matching_type=matching,
                                   do_verbose=verbose)
                elif method == 'GMDE':
                    solution = gmde(points=points,
                                    size=population_size,
                                    max_iteration=max_iteration,
                                    n_clusters=cluster_seq,
                                    matching_type=matching,
                                    do_verbose=verbose)
                elif method == 'DMDE':
                    solution = dmde(points=points,
                                    size=population_size,
                                    max_iteration=max_iteration,
                                    n_clusters=cluster_seq,
                                    matching_type=matching,
                                    do_verbose=verbose)
                elif method == 'HMDE':
                    solution = hmde(points=points,
                                    size=population_size,
                                    max_iteration=max_iteration,
                                    n_clusters=cluster_seq,
                                    matching_type=matching,
                                    do_verbose=verbose)

                execution_time = time.time() - start_time

                utils.print_result(round(solution.get_score(), 4), KMeans.get_score(), round(execution_time, 2))
