import sys


def dmde_selection(population, candidate):
    candidate_score = candidate.get_score()
    index_near = -1
    min_diff = sys.float_info.max
    for index, solution in enumerate(population.solutions):
        actual_diff = abs(solution.get_score() - candidate_score)
        if actual_diff < min_diff:
            min_diff = actual_diff
            index_near = index
    if index_near == -1:
        raise ValueError('Error during selection phase')
    return index_near
