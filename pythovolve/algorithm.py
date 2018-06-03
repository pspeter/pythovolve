import random
import time
from abc import ABCMeta, abstractmethod
from multiprocessing import Queue, Process
from typing import Tuple, List, Sequence

from pythovolve.callbacks import Callback
from pythovolve.crossover import Crossover
from pythovolve.individuals import Individual
from pythovolve.mutation import Mutator
from pythovolve.plotting import TSPPlot, ProgressPlot
from pythovolve.problems import Problem, TravellingSalesman
from pythovolve.selection import Selector


class EvolutionAlgorithm(metaclass=ABCMeta):
    def __init__(self, problem: Problem,
                 population_size: int = 100,
                 max_generations: int = 1000,
                 callbacks: Sequence[Callback] = None,
                 plot_progress: bool = False,
                 verbosity: int = 1,
                 **_):
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
        self.population = [self.problem.create_individual() for _ in range(self.population_size)]

        self.best_scores = []
        self.current_best_scores = []
        self.callbacks = callbacks or []
        for callback in self.callbacks:
            callback.subscribe(self)

        # Note: interactive plotting has only been tested with backend TkAgg and
        # does definitely not work in Pycharm's SciView as of version 2018.1.4
        self.plot_progress = plot_progress
        self.verbosity = verbosity

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
                for callback in self.callbacks:
                    callback.on_train_end()

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
        data_queue = Queue()

        if isinstance(self.problem, TravellingSalesman):
            plot_process = Process(target=TSPPlot, args=(self.max_generations, self.problem, data_queue))
        else:
            plot_process = Process(target=ProgressPlot, args=(self.max_generations, data_queue))

        return data_queue, plot_process


    @classmethod
    def from_args(cls, **kwargs):
        cls.from_args(**kwargs)


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
                 plot_progress: bool = False,
                 verbosity: int = 1,
                 **kwargs):
        super().__init__(problem, population_size, max_generations, callbacks, plot_progress, verbosity, **kwargs)
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
                 plot_progress: bool = False,
                 verbosity: int = 1,
                 **kwargs):
        super().__init__(problem, selector, crossover, mutator, population_size,
                         num_elites, max_generations, callbacks, plot_progress, verbosity, **kwargs)
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
        return chosen

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
                 min_sigma: float = None,
                 callbacks: Sequence[Callback] = None,
                 plot_progress: bool = False,
                 verbosity: int = 1, **kwargs):
        if num_children > population_size:
            raise ValueError("Number of children larger than number of parents")

        super().__init__(problem, population_size, max_generations, callbacks, plot_progress, verbosity, **kwargs)
        self.sigma_multiplier = sigma_multiplier
        self.keep_parents = keep_parents
        self.sigma = sigma_start
        self.num_children = num_children
        self.selector = selector
        self.mutator = mutator
        self.min_sigma = min_sigma

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

        if self.generation >= self.max_generations or self.min_sigma and self.sigma < self.min_sigma:
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
