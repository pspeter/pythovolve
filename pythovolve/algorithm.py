import random
from abc import ABCMeta, abstractmethod
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Tuple, List, Sequence, Any
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
                 callbacks: Sequence[Callback] = None,
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
            self.best = self.current_best

    def evolve(self) -> None:
        for callback in self.callbacks:
            callback.on_train_start()

        self.stop_evolving = False
        if self.plot_progress:
            data_queue, plot_process = self._start_plot_process()
            try:
                plot_process.start()
                time.sleep(2)  # wait for figure to open
                while not self.stop_evolving:
                    self.evolve_once()
                    data_queue.put((self.current_best.score, self.best))
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

    def _start_plot_process(self) -> Tuple[Queue, Process]:
        if isinstance(self.problem, TravellingSalesman):
            data_queue = Queue()
            plot_process = Process(target=TSPPlot, args=(self.max_generations, self.problem, data_queue))
        else:
            data_queue = Queue()
            plot_process = Process(target=ProgressPlot, args=(self.max_generations, data_queue))
            return data_queue, plot_process
        return data_queue, plot_process


class GeneticAlgorithm(EvolutionAlgorithm):
    """Genetic Algorithm (GA) implementation.

    :param problem: Problem that can create and score individuals
    :param selector: Callable that returns one individual from a population
    :param crossover: Callable that crossbreeds two individuals and returns two children
    :param mutator: Callable that mutates an individual
    :param population_size: The 'mu' parameter. Number of parents for each generation
    :param num_elites: How many of the best individuals should be kept for the next generation
    :param max_generations: Stops after that many generations
    :param callbacks: Optional callbacks
    :param plot_progress: Wether to plot the progress while running. This has only been
        tested with the matplotlib backend "TkAgg" and is not garantueed to work with others.
    """

    def __init__(self, problem: Problem, selector: Selector,
                 crossover: Crossover, mutator: Mutator,
                 population_size: int = 100, num_elites: int = 0,
                 max_generations: int = 1000,
                 callbacks: Sequence[Callback] = None,
                 plot_progress: bool = False):
        super().__init__(problem, population_size, max_generations, callbacks, plot_progress)
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator

        self.num_elites = num_elites

    def evolve_once(self) -> None:
        for callback in self.callbacks:
            callback.on_generation_start()

        elites, non_elites = self._split_elites(self.population)

        children = self._generate_children()

        # we don't want to mutate our elites, only the children
        self.population = [self.mutator(child) for child in children] + elites

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

        # in case population_size is not an even number, add one more child
        if len(children) < self.population_size:
            father, mother = self.selector(self.population), self.selector(self.population)
            children += self.crossover(father, mother)[:1]

        return children


class OSGeneticAlgorithm(GeneticAlgorithm):
    """Genetic Algorithm (GA) implementation using offspring selection (OS) as described by
        Affenzeller M., Wagner S. (2005)

    """

    def __init__(self, problem: Problem, selector: Selector,
                 crossover: Crossover, mutator: Mutator,
                 population_size: int = 100, num_elites: int = 0,
                 max_generations: int = 1000, max_selection_pressure: float = 10,
                 success_ratio: float = 0.5,
                 comparison_factor_bounds: Tuple[float, float] = (0., 1.),
                 callbacks: Sequence[Callback] = None,
                 plot_progress: bool = False):
        super().__init__(problem, selector, crossover, mutator, population_size,
                         num_elites, max_generations, callbacks, plot_progress)
        self.comparison_factor_bounds = comparison_factor_bounds
        self.comparison_factor = comparison_factor_bounds[0]
        self._comparison_factor_increase = (comparison_factor_bounds[1] -
                                            comparison_factor_bounds[0]) / self.max_generations
        self.max_selection_pressure = max_selection_pressure
        self.success_ratio = success_ratio
        self.selection_pressure = 0

    def _generate_children(self):
        success_children = []
        failure_children = []

        while len(success_children) < self.success_ratio * self.population_size:
            father, mother = self.selector(self.population), self.selector(self.population)
            child1, child2 = self.crossover(father, mother)

            if self._is_successful(child1, father, mother):
                success_children.append(child1)
            else:
                failure_children.append(child1)

            if self._is_successful(child2, father, mother):
                success_children.append(child2)
            else:
                failure_children.append(child2)

            self.selection_pressure = (len(failure_children) + len(success_children)
                                       + self.population_size) / self.population_size

            if self.selection_pressure > self.max_selection_pressure:
                print("Selection pressure too high. Stopping...")
                self.stop_evolving = True
                break

        self.comparison_factor += self._comparison_factor_increase

        # if success ratio was reached before enough children were produced for a full
        # new generation, create more children
        while len(failure_children) + len(success_children) < self.population_size:
            father, mother = self.selector(self.population), self.selector(self.population)
            failure_children.extend(self.crossover(father, mother))

        chosen = success_children + random.sample(failure_children, k=self.population_size - len(success_children))
        return [self.mutator(child) for child in chosen]

    def _is_successful(self, child: Individual, parent1: Individual, parent2: Individual) -> bool:
        self.problem.score_individual(child)
        parents = sorted([parent1, parent2])
        limit = parents[0].score * self.comparison_factor + parents[1].score * (1 - self.comparison_factor)
        return child.score < limit


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
                 callbacks: Sequence[Callback] = None,
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

        self.best_scores.append(self.best.score)
        self.current_best_scores.append(self.current_best.score)
        self.generation += 1

        if self.generation >= self.max_generations:
            self.stop_evolving = True

        for callback in self.callbacks:
            callback.on_generation_end()

    def _adapt_sigma(self, num_success: int):
        """Sigma adaption as suggested by Schwefel (1981)"""
        if num_success > 1 / 5 * self.num_children:
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
    def __init__(self, max_generations: int, data_queue: Queue, fig: Figure = None,
                 axes: Sequence[Axes] = None):
        self.max_generations = max_generations
        # set to True if there's new data to plot, set to false after it has been plotted
        self.stale = False

        if fig is not None and axes is not None:
            self.fig = fig
            self.axes = axes
        else:
            self.fig, axes = plt.subplots()
            self.axes = [axes]

        self.data_queue = data_queue
        self.current_best_scores = []
        self.best_scores = []
        self.generation = 0

        self.total_line, = self.axes[0].plot([], [], 'r-', animated=True, label="Total best")
        self.current_line, = self.axes[0].plot([], [], 'g.', animated=True, label="Generation best")

        # setup the animation
        self.animation = FuncAnimation(self.fig, self._update, init_func=self._init,
                                       blit=True, interval=1000 // 20)

        self.legend = self.axes[0].legend()

        sns.set()
        plt.show()

    def _get_newest_data(self):
        while not self.data_queue.empty():
            current_best_score, best = self.data_queue.get()
            self.current_best_scores.append((self.generation, current_best_score))
            self.best_scores.append((self.generation, best.score))
            self.best = best
            self.generation += 1
            self.stale = True

    def _init(self):
        self._get_newest_data()

        if len(self.best_scores) > 0:
            x_max = min(self.max_generations - 1, int(len(self.best_scores) * 2 + 10))
            y_max = self.best.score * 1.2
        else:
            x_max = 100
            y_max = 1e-5

        ax = self.axes[0]
        ax.set_title("Progress")
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Score")

        self.legend = ax.legend()

        return self.current_line, self.total_line, self.legend

    def _update(self, _):
        self._get_newest_data()

        if len(self.best_scores) > 0 and self.stale:
            ax = self.axes[0]

            # update range of x-axis
            _, x_max = ax.get_xlim()
            if len(self.best_scores) + 1 > x_max * 0.95 and not x_max == self.max_generations - 1:
                ax.set_xlim(0, min(self.max_generations - 1, int(x_max * 2 + 10)))
                ax.figure.canvas.draw()

            # update range of y-axis
            _, y_max = ax.get_ylim()
            if self.best.score > y_max * 0.95:
                ax.set_ylim(0, self.best.score * 1.3)
                ax.figure.canvas.draw()

            self.current_line.set_data(zip(*sorted(self.current_best_scores)))
            self.total_line.set_data(zip(*sorted(self.best_scores)))

        return self.current_line, self.total_line, self.legend


class TSPPlot(ProgressPlot):
    def __init__(self, max_generations: int, problem: TravellingSalesman, data_queue: Queue):
        self.problem = problem
        self.path_lines = []
        self.city_points = []

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        super().__init__(max_generations, data_queue, fig, axes)

    def _init(self):
        super()._init()
        ax = self.axes[1]
        area = self.problem.defined_area

        ax.set_title("Best Solution")
        ax.set_xlim(area.min.x, area.max.x)
        ax.set_ylim(area.min.y, area.max.y)

        for point in self.city_points:
            del point

        x_cities = [city.x for city in self.problem.cities]
        y_cities = [city.y for city in self.problem.cities]
        self.city_points = ax.plot(x_cities, y_cities, ls="", marker="*", label="Cities")

        self._plot_paths()

        return (self.current_line, self.total_line, self.legend, *self.path_lines)

    def _update(self, frame):
        super()._update(frame)
        self._plot_paths()
        return (self.current_line, self.total_line, self.legend, *self.path_lines)

    def _plot_paths(self):
        if not self.best_scores or not self.stale:
            return

        ax = self.axes[1]

        while self.path_lines:
            # to completely get rid of the lines, this is necessary
            # see https://stackoverflow.com/questions/4981815
            self.path_lines.pop(0).remove()

        path = [self.problem.cities[idx] for idx in self.best.phenotype]
        self.path_lines = ax.plot([path[-1].x, path[0].x], [path[-1].y, path[0].y], "k-")

        for start, dest in zip(path, path[1:]):
            self.path_lines += ax.plot([dest.x, start.x], [dest.y, start.y], "k-")


if __name__ == "__main__":
    random.seed(123)

    n_cities = 60
    tsp = TravellingSalesman.create_random(n_cities)
    #mut = multi_path_mutator
    #cx = multi_crossover
    sel = multi_selector

    # prob = goldstein_price_problem
    # print("Problem:", prob.expression)
    # print("Best known so far:", prob.best_known)
    mut = RealValueMutator(1)
    cx = single_point_crossover
    # sel = multi_selector

    # es = EvolutionStrategy(tsp, sel, mut, sigma_start=0.5,
    #                        keep_parents=False, max_generations=500, plot_progress=True)
    # es.evolve()
    # print("best:", es.best)

    # ga = GeneticAlgorithm(prob, sel, cx, mut, 100, 3, max_generations=100, plot_progress=True)
    # ga.evolve()
    # print("best:", ga.best)
    start = time.time()
    ga = OSGeneticAlgorithm(goldstein_price_problem, sel, cx, mut, 200, 1, comparison_factor_bounds=(0.5, 1),
                            max_generations=100, plot_progress=False)
    ga.evolve()
    print("OSGA best found: ", ga.best, "in", time.time() - start, "seconds")
    start = time.time()
    ga = GeneticAlgorithm(goldstein_price_problem, sel, cx, mut, 200, 1, max_generations=500, plot_progress=False)
    ga.evolve()
    print("GA best found: ", ga.best, "in", time.time() - start, "seconds")
