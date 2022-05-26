import sys


def dmde_selection(population, candidate):
    candidate_score = candidate.get_score()
    index_near = -1
    score_near = sys.float_info.max
    for index, solution in enumerate(population.solutions):
        actual_score = abs(solution.get_score() - candidate_score)
        if actual_score < score_near:
            score_near = actual_score
            index_near = index
    if index_near == -1:
        raise ValueError('Error during selection phase')
    if candidate_score < population.get_solution(index_near).get_score():
        population.replace_solution(index_near, candidate)
