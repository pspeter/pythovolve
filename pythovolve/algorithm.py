import random
from typing import Tuple, List

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pythovolve.callbacks import EarlyStopper, Callback
from pythovolve.problems import Problem, TravellingSalesman
from pythovolve.individuals import Individual, PathIndividual
from pythovolve.crossover import CycleCrossover, Crossover
from pythovolve.selection import Selector, ProportionalSelector
from pythovolve.mutation import Mutator, TranslocationMutator, InversionMutator


class GeneticAlgorithm:
    def __init__(self, problem: Problem, population: List[Individual],
                 selector: Selector, crossover: Crossover,
                 mutator: Mutator, num_elites: int = 0,
                 use_offspring_selection: bool = False,
                 max_generations=1000,
                 callbacks: List[Callback] = None,
                 plot_progress: bool = False):
        self._population: List[Individual] = None
        self.best: Individual = None
        self.current_best: Individual = None

        self.stop_evolving = False
        self.generation = 0

        self.problem = problem
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator
        self.population_size = len(population)
        self.max_generations = max_generations

        if self.population_size == 0:
            raise ValueError("Initial population is empty")

        self.population = population
        self.num_elites = num_elites
        self.use_offspring_selection = use_offspring_selection  # todo

        self.callbacks = callbacks or []

        self.plot_progress = plot_progress

    @property
    def population(self) -> List[Individual]:
        return self._population

    @population.setter
    def population(self, population: List[Individual]):
        self._population = population

        for individual in self.population:
            self.problem.score_individual(individual)

        self.current_best = sorted(self.population)[-1]

        if not self.best or self.current_best > self.best:
            print("new best:", self.current_best.score)
            self.best = self.current_best

    def evolve(self) -> None:
        for callback in self.callbacks:
            callback.on_train_start()

        self.stop_evolving = False
        if self.plot_progress:
            progress_plot = ProgressPlot(self)
            progress_plot.start_animation()
        else:
            while not self.stop_evolving:
                self.evolve_once()

        for callback in self.callbacks:
            callback.on_train_end()

    def evolve_once(self) -> None:
        for callback in self.callbacks:
            callback.on_generation_start()

        elites, non_elites = self._split_elites(self.population)
        children = []

        for _ in range(self.population_size // 2):
            children += self.crossover(self.selector(self.population), self.selector(self.population))

        # in case the above range() was rounded down, add one more child
        if len(children) < self.population_size:
            children += self.crossover(self.selector(self.population), self.selector(self.population))[0]

        # we don't want to mutate our elites, only the children
        self.population = [self.mutator(child) for child in children] + elites
        self.generation += 1
        if self.generation >= self.max_generations:
            self.stop_evolving = True

        for callback in self.callbacks:
            callback.on_generation_end()

    def _split_elites(self, population: List[Individual]) -> Tuple[List[Individual], List[Individual]]:
        """
        :param population: Population to be split into elites and non-elites
        :return: (elites, non-elites)
        """
        if self.num_elites == 0:
            return [], population
        sorted_population = sorted(population)
        return sorted_population[-self.num_elites:], sorted_population[:-self.num_elites]


class ProgressPlot:
    def __init__(self, algorithm: GeneticAlgorithm):
        self.algorithm = algorithm

        # set up the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, .0015)  # todo
        self.line, = plt.plot([], [], 'r-', animated=True)

        self.best = []
        self.gens = []

        # setup the animation
        self.animation = FuncAnimation(self.fig, self._update, blit=True, interval=1)

    def _update(self, gen):
        _, x_max = self.ax.get_xlim()

        if gen + 1 > x_max * 0.95:
            self.ax.set_xlim(0, int(x_max * 1.3 + 10))
            self.ax.figure.canvas.draw()

        _, y_max = self.ax.get_ylim()

        if self.algorithm.best.score > y_max * 0.95:
            self.ax.set_ylim(0, self.algorithm.best.score * 1.3)
            self.ax.figure.canvas.draw()

        self.gens.append(gen)
        self.best.append(self.algorithm.best.score)
        self.algorithm.evolve_once()

        self.line.set_data(self.gens, self.best)


        return self.line,

    def stop(self):
        self.animation.event_source.stop()

    def start_animation(self):
        plt.show()


if __name__ == "__main__":
    random.seed(123)
    n_cities = 30
    tsp = TravellingSalesman.create_random(n_cities)
    mut = InversionMutator(0.15)
    cx = CycleCrossover()
    sel = ProportionalSelector()
    pop = [PathIndividual.create_random(n_cities) for _ in range(100)]
    ga = GeneticAlgorithm(tsp, pop, sel, cx, mut, 3, plot_progress=True)
    ga.evolve()
    print("best found: ", tsp.best_known.score)  # best found:  0.0021516681121798863
