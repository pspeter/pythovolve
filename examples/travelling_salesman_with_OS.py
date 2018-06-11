import random

from pythovolve.algorithm import OSGeneticAlgorithm
from pythovolve.callbacks import TimerCallback, EarlyStopCallback
from pythovolve.crossover import order_crossover, cycle_crossover
from pythovolve.mutation import InversionMutator, TranslocationMutator
from pythovolve.problems import TravellingSalesman
from pythovolve.selection import TournamentSelector, linear_rank_selector

random.seed(123)

if __name__ == "__main__":
    problem = TravellingSalesman.create_random(150)

    selectors = [TournamentSelector(15), TournamentSelector(3), linear_rank_selector]
    crossovers = [order_crossover, cycle_crossover]
    mutators = [TranslocationMutator(0.7), InversionMutator(0.7)]

    callbacks = [TimerCallback(),
                 EarlyStopCallback(max_no_progress=500, max_seconds=90)]

    ga = OSGeneticAlgorithm(problem, selectors, crossovers, mutators,
                            population_size=100, num_elites=3,
                            max_generations=10000,
                            max_selection_pressure=30,
                            success_ratio=0.2,
                            callbacks=callbacks, plot_progress=True)

    ga.evolve()

    print("Best solution:", ga.best.score)
