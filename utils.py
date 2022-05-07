from matplotlib import pyplot as plt


def show_solution(points, solution):
    plt.scatter(points['x'], points['y'], c=solution.membership_vector, cmap='rainbow')
    plt.show()