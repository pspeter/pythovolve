import random

from pythovolve.algorithm import GeneticAlgorithm
from pythovolve.callbacks import TimerCallback, EarlyStopCallback
from pythovolve.crossover import order_crossover
from pythovolve.mutation import inversion_mutator, translocation_mutator
from pythovolve.problems import TravellingSalesman
from pythovolve.selection import TournamentSelector

random.seed(123)

if __name__ == "__main__":
    problem = TravellingSalesman.create_random(40)

    selectors = [TournamentSelector(10)]
    crossovers = [order_crossover]
    mutators = [translocation_mutator, inversion_mutator]

    callbacks = [TimerCallback(),
                 EarlyStopCallback(max_no_progress=500, max_seconds=20)]

    ga = GeneticAlgorithm(problem, selectors, crossovers, mutators,
                          population_size=50, num_elites=1,
                          max_generations=10000,
                          callbacks=callbacks, plot_progress=True)

    ga.evolve()

    print("Best solution:", ga.best.score)
