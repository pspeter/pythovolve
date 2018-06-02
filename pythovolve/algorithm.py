import random
from abc import ABCMeta, abstractmethod
from multiprocessing import Queue, Process
from typing import Tuple, List
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

from pythovolve.callbacks import Callback
from pythovolve.problems import Problem, TravellingSalesman, MultiDimFunction, sphere_problem, goldstein_price_problem, \
    booth_problem
from pythovolve.individuals import Individual, PathIndividual
from pythovolve.crossover import Crossover, CycleCrossover, order_crossover, cycle_crossover, multi_crossover, \
    single_point_crossover
from pythovolve.selection import Selector, ProportionalSelector, TournamentSelector, LinearRankSelector, multi_selector
from pythovolve.mutation import Mutator, InversionMutator, multi_path_mutator, real_value_mutator, RealValueMutator


class EvolutionAlgorithm(metaclass=ABCMeta):
    def __init__(self, problem: Problem,
                 population_size: int = 100,
                 max_generations: int = 1000,
                 callbacks: List[Callback] = None,
                 plot_progress: bool = False):
        self.best: Individual = None
        self.current_best: Individual = None

        self.stop_evolving = False
        self.generation = 0

        self.problem = problem
        self.max_generations = max_generations
        self.population_size = population_size

        if self.population_size == 0:
            raise ValueError("Initial population is empty")

        self._population = None
        self.population = [self.problem.create_individual() for _ in range(300)]

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

        self.current_best = min(self.population)

        if self.best is None or self.current_best < self.best:
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
                time.sleep(2)  # wait for figure to open
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

    @abstractmethod
    def evolve_once(self) -> None:
        pass


class GeneticAlgorithm(EvolutionAlgorithm):
    """Genetic Algorithm (GA) implementation.

    :param problem: Problem that can create and score individuals
    :param selector: Callable that returns one individual from a population
    :param crossover: Callable that crossbreeds two individuals and returns two children
    :param mutator: Callable that mutates an individual
    :param population_size: The 'mu' parameter. Number of parents for each generation
    :param num_elites: How many of the best individuals should be kept for the next generation
    :param use_offspring_selection: Wether to use offspring selection (OS) as described by
        Affenzeller M., Wagner S. (2005)
    :param max_generations: Stops after that many generations
    :param callbacks: Optional callbacks
    :param plot_progress: Wether to plot the progress while running. This has only been
        tested with the matplotlib backend "TkAgg" and is not garantueed to work with others.
    """
    def __init__(self, problem: Problem, selector: Selector,
                 crossover: Crossover, mutator: Mutator,
                 population_size: int = 100, num_elites: int = 0,
                 use_offspring_selection: bool = False,
                 max_generations: int = 1000,
                 callbacks: List[Callback] = None,
                 plot_progress: bool = False):
        super().__init__(problem, population_size, max_generations, callbacks, plot_progress)
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator

        self.num_elites = num_elites
        self.use_offspring_selection = use_offspring_selection  # todo

    def evolve_once(self) -> None:
        for callback in self.callbacks:
            callback.on_generation_start()

        elites, non_elites = self._split_elites(self.population)

        children = self._generate_children()

        # we don't want to mutate our elites, only the children
        self.population = [self.mutator(child, 0.5) for child in children] + elites

        self.generations.append(self.generation)
        self.best_scores.append(self.best.score)
        self.current_best_scores.append(self.current_best.score)

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
        return sorted_population[:self.num_elites], sorted_population[self.num_elites:]

    def _generate_children(self):
        children = []

        for _ in range(self.population_size // 2):
            father, mother = self.selector(self.population), self.selector(self.population)
            children += self.crossover(father, mother)

        # in case the above range() was rounded down, add one more child
        if len(children) < self.population_size:
            father, mother = self.selector(self.population), self.selector(self.population)
            children += self.crossover(father, mother)[:1]

        return children


class EvolutionStrategy(EvolutionAlgorithm):
    """Evolution strategy (ES) implementation.

    :param problem: Problem that can create and score individuals
    :param selector: Callable that returns one individual from a population
    :param mutator: Callable that mutates an individual
    :param population_size: The 'mu' parameter. Number of parents for each generation
    :param num_children: The 'lambda' parameter. Number of children for each generation
    :param sigma_start: Initital sigma value. Sigma controls the strength of mutation
    :param keep_parents: Wether to use 'mu+lambda' strategy (True) or 'mu,lambda' (False)
    :param sigma_multiplier: How much selection pressure should control sigma
    :param max_generations: Stops after that many generations
    :param callbacks: Optional callbacks
    :param plot_progress: Wether to plot the progress while running. This has only been
        tested with the matplotlib backend "TkAgg" and is not garantueed to work with others.
    """
    def __init__(self, problem: Problem, selector: Selector,
                 mutator: Mutator,
                 population_size: int = 100, num_children: int = 10,
                 sigma_start: float = 1., keep_parents: bool = False,
                 sigma_multiplier: float = 1.15,
                 max_generations: int = 1000,
                 callbacks: List[Callback] = None,
                 plot_progress: bool = False):

        if num_children > population_size:
            raise ValueError("Number of children larger than number of parents")

        super().__init__(problem, population_size, max_generations, callbacks, plot_progress)
        self.sigma_multiplier = sigma_multiplier
        self.keep_parents = keep_parents
        self.sigma = sigma_start
        self.num_children = num_children
        self.selector = selector
        self.mutator = mutator

    def evolve_once(self) -> None:
        for callback in self.callbacks:
            callback.on_generation_start()

        children, num_success = self._generate_children()

        if self.keep_parents:
            children.extend(self.population)

        self.population = sorted(children)[:self.population_size]  # selection

        self._adapt_sigma(num_success)

        self.generations.append(self.generation)
        self.best_scores.append(self.best.score)
        self.current_best_scores.append(self.current_best.score)
        self.generation += 1

        if self.generation >= self.max_generations:
            self.stop_evolving = True

        for callback in self.callbacks:
            callback.on_generation_end()

    def _adapt_sigma(self, num_success: int):
        """Sigma adaption as suggested by Schwefel (1981)"""
        if num_success > 1/5 * self.num_children:
            self.sigma *= self.sigma_multiplier
        else:
            self.sigma /= self.sigma_multiplier

    def _generate_children(self):
        children = []
        num_success = 0

        for _ in range(self.num_children):
            parent = self.selector(self.population)
            child = self.mutator(parent.clone(), self.sigma)
            self.problem.score_individual(child)
            children.append(child)

            if child.score < parent.score:
                num_success += 1

        return children, num_success


class ProgressPlot:
    def __init__(self, max_generations: int, data_queue: Queue):
        self.max_generations = max_generations
        self.data_queue = data_queue
        self.data = None

        # set up the plot
        sns.set()
        self.fig, self.ax = plt.subplots()
        self.total_line, = plt.plot([], [], 'r-', animated=True, label="Total best")
        self.current_line, = plt.plot([], [], 'g*', animated=True, label="Generation best")

        # setup the animation
        self.animation = FuncAnimation(self.fig, self._update, init_func=self._init,
                                       blit=True, interval=1000 // 24)

        self.legend = plt.legend()
        plt.show()

    def _init(self):  # todo fix redraw when queue is empty
        self._get_newest_data()
        if self.data:
            x_values, current_best, total_best = self.data
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
        self._get_newest_data()

        if self.data:
            x_values, current_best, total_best = self.data

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

    def _get_newest_data(self):
        # get newest result
        while not self.data_queue.empty():
            self.data = self.data_queue.get()


if __name__ == "__main__":
    random.seed(123)
    prob = goldstein_price_problem
    print("Problem:", prob.expression)
    print("Best known so far:", prob.best_known)
    mut = RealValueMutator(1)
    cx = single_point_crossover
    sel = multi_selector

    es = EvolutionStrategy(prob, sel, mut, sigma_start=0.5,
                           keep_parents=False, max_generations=100, plot_progress=True)
    es.evolve()
    print("best:", es.best)

    # ga = GeneticAlgorithm(prob, sel, cx, mut, 100, 3, max_generations=100, plot_progress=True)
    # ga.evolve()
    # print("best:", ga.best)

    # n_cities = 130
    # tsp = TravellingSalesman.create_random(n_cities)
    # mut = multi_path_mutator
    # cx = multi_crossover
    # sel = multi_selector
    # import time
    #
    # start = time.time()
    # ga = GeneticAlgorithm(tsp, sel, cx, mut, 100, 20, max_generations=2500, plot_progress=True)
    # ga.evolve()
    # print("time: ", time.time() - start)
    # print("best found: ", tsp.best_known.score)
