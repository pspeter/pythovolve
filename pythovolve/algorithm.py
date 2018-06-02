import random
from multiprocessing import Queue, Process
from typing import Tuple, List

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

from pythovolve.callbacks import Callback
from pythovolve.problems import Problem, TravellingSalesman
from pythovolve.individuals import Individual, PathIndividual
from pythovolve.crossover import Crossover, CycleCrossover, order_crossover, cycle_crossover, multi_crossover
from pythovolve.selection import Selector, ProportionalSelector, TournamentSelector, LinearRankSelector, multi_selector
from pythovolve.mutation import Mutator, InversionMutator, multi_mutator


class GeneticAlgorithm:
    def __init__(self, problem: Problem, selector: Selector,
                 crossover: Crossover, mutator: Mutator,
                 population_size: int = 100, num_elites: int = 0,
                 use_offspring_selection: bool = False,
                 max_generations: int = 1000,
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
        self.max_generations = max_generations
        self.population_size = population_size

        if self.population_size == 0:
            raise ValueError("Initial population is empty")

        self.population = [self.problem.create_individual("path") for _ in range(300)]
        self.num_elites = num_elites
        self.use_offspring_selection = use_offspring_selection  # todo

        self.best_scores = []
        self.current_best_scores = []
        self.generations = []
        self.callbacks = callbacks or []

        # Note: interactive plotting has only been tested with backend TkAgg and
        # does definitely not work in Pycharm's SciView as of version 2018.1.4
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
            data_queue = Queue()
            plot_process = Process(target=ProgressPlot, args=(self.max_generations, data_queue))
            try:
                plot_process.start()
                time.sleep(1)  # wait for figure to open
                while not self.stop_evolving:
                    self.evolve_once()
                    data_queue.put((self.generations, self.current_best_scores, self.best_scores))
            finally:
                plot_process.join()

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
            father, mother = self.selector(self.population), self.selector(self.population)
            children += self.crossover(father, mother)

        # in case the above range() was rounded down, add one more child
        if len(children) < self.population_size:
            children += self.crossover(self.selector(self.population), self.selector(self.population))[0]

        # we don't want to mutate our elites, only the children
        self.population = [self.mutator(child) for child in children] + elites

        self.generations.append(self.generation)
        self.best_scores.append(1 / self.best.score)
        self.current_best_scores.append(1 / self.current_best.score)

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
    def __init__(self, max_generations: int, data_queue: Queue):
        self.max_generations = max_generations
        self.data_queue = data_queue
        # set up the plot
        sns.set()
        self.fig, self.ax = plt.subplots()
        self.total_line, = plt.plot([], [], 'r-', animated=True, label="Total best")
        self.current_line, = plt.plot([], [], 'b:', animated=True, label="Generation best")

        # setup the animation
        self.animation = FuncAnimation(self.fig, self._update, init_func=self._init,
                                       blit=True, interval=1000 // 24)

        self.legend = plt.legend()
        plt.show()

    def _init(self):
        if not self.data_queue.empty():
            x_values, current_best, total_best = self.data_queue.get()
            x_max = min(self.max_generations - 1, int(len(x_values) * 2 + 10))
            y_max = max(total_best) * 1.2
        else:
            x_max = 100
            y_max = 1e-5

        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Score")

        return self.current_line, self.total_line, self.legend

    def _update(self, _):
        if not self.data_queue.empty():
            x_values, current_best, total_best = self.data_queue.get()

            # get newest result
            while not self.data_queue.empty():
                x_values, current_best, total_best = self.data_queue.get()

            # update range of x-axis
            _, x_max = self.ax.get_xlim()
            # if x_values[-1] + 1 >= x_max and not x_max == self.max_generations - 1:
            #     self.ax.set_xlim(0, x_max + 1)
            if x_values[-1] + 1 > x_max * 0.95 and not x_max == self.max_generations - 1:
                self.ax.set_xlim(0, min(self.max_generations - 1, int(x_max * 2 + 10)))
                self.ax.figure.canvas.draw()

            # update range of y-axis
            _, y_max = self.ax.get_ylim()
            if max(total_best) > y_max * 0.95:
                self.ax.set_ylim(0, max(total_best) * 1.3)
                self.ax.figure.canvas.draw()

            self.current_line.set_data(x_values, current_best)
            self.total_line.set_data(x_values, total_best)

        return self.current_line, self.total_line, self.legend


if __name__ == "__main__":
    random.seed(123)
    n_cities = 130
    tsp = TravellingSalesman.create_random(n_cities)
    mut = multi_mutator
    cx = multi_crossover
    sel = multi_selector
    import time

    start = time.time()
    ga = GeneticAlgorithm(tsp, 300, sel, cx, mut, 20, max_generations=2500, plot_progress=True)
    ga.evolve()
    print("time: ", time.time() - start)
    print("best found: ", tsp.best_known.score)
