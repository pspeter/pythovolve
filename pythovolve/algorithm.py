import random
from abc import ABCMeta, abstractmethod
from typing import Tuple, List

from matplotlib.animation import FuncAnimation

from pythovolve.problems import Problem, TravellingSalesman, City
from .individuals import Individual, BinaryIndividual, PathIndividual


class Mutator(metaclass=ABCMeta):
    def __init__(self, probability: float = 0.1):
        self.probability = probability

    @abstractmethod
    def __call__(self, individual: Individual) -> Individual:
        pass


class BitFlipMutator(Mutator):
    def __init__(self, max_flips: int, probability: float = 0.1):
        super().__init__(probability)
        self.max_flips = max_flips

    def __call__(self, individual: BinaryIndividual) -> BinaryIndividual:
        if random.random() > self.probability:
            return individual

        len_pheno = len(individual.phenotype)
        num_flips = random.randint(1, max(len_pheno, self.max_flips))
        to_flip = random.sample(range(len_pheno), num_flips)

        for flip_index in to_flip:
            individual.phenotype[flip_index] ^= True  # flips bit

        return individual


class InversionMutator(Mutator):
    def __call__(self, individual: PathIndividual) -> PathIndividual:
        if random.random() > self.probability:
            return individual

        path = individual.phenotype
        start, end = random.sample(list(range(len(path))), 2)
        path[start:end] = reversed(path[start:end])
        return individual


class TranslocationMutator(Mutator):
    def __call__(self, individual: PathIndividual) -> PathIndividual:
        if random.random() > self.probability:
            return individual

        path = individual.phenotype
        start, mid, end = random.sample(list(range(len(path))), 3)
        path[start:mid] = reversed(path[start:mid])
        path[mid:end] = reversed(path[mid:end])
        return individual


class Crossover(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, parents: Tuple[Individual, Individual]) -> Tuple[Individual, Individual]:
        pass


class CycleCrossover(Crossover):
    def __call__(self, parents: Tuple[PathIndividual, PathIndividual]) \
            -> Tuple[PathIndividual, PathIndividual]:
        first = random.randrange(len(parents[0].phenotype))
        return self._create_child(*parents, first), self._create_child(*reversed(parents), first)

    @staticmethod
    def _create_child(parent1: PathIndividual, parent2: PathIndividual,
                      first_index: int) -> PathIndividual:
        child = parent2.phenotype[:]
        child[first_index] = parent1.phenotype[first_index]
        current = parent1.phenotype.index(parent2.phenotype[first_index])

        while current != first_index:
            child[current] = parent1.phenotype[current]
            current = parent1.phenotype.index(parent2.phenotype[current])

        return PathIndividual(child)


class Selector(metaclass=ABCMeta):
    def __init__(self, num_select: int):
        self.num_select = num_select

    @abstractmethod
    def __call__(self, population: List[Individual]) -> Tuple[List[Individual], List[Individual]]:
        pass


class ProportionalSelector(Selector):
    def __init__(self, num_select: int):
        super().__init__(num_select)

    def __call__(self, population: List[Individual]) -> Tuple[List[Individual], List[Individual]]:
        if len(population) < self.num_select:
            raise ValueError("Population smaller than num_select")
        weights = [indiv.score for indiv in population]
        return random.choices(population, weights, k=self.num_select)


class GeneticAlgorithm:
    def __init__(self, problem: Problem, population: List[Individual],
                 selector: Selector, crossover: Crossover,
                 mutator: Mutator, num_elites: int = 0,
                 use_offspring_selection: bool = False):
        self._population: List[Individual] = None
        self.best: Individual = None
        self.current_best: Individual = None
        self.problem = problem
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator
        self.population_size = len(population)
        self.population = population
        self.num_elites = num_elites
        self.use_offspring_selection = use_offspring_selection

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
        elites, non_elites = self._split_elites(self.population)
        selected = self.selector(non_elites)
        children = []

        for _ in range((self.population_size - self.num_elites) // 2):
            children.extend(self.crossover(random.sample(selected, 2)))

        # in case the above range() was rounded down, add one more child
        if len(children) < self.population_size:
            children.append(self.crossover(random.sample(selected, 2))[0])

        self.population = [self.mutator(child) for child in children] + elites

    def _split_elites(self, population: List[Individual]) -> Tuple[List[Individual], List[Individual]]:
        """
        :param population: Population to be split into elites and non-elites
        :return: (elites, non-elites)
        """
        if self.num_elites == 0:
            return [], population
        sorted_population = sorted(population)
        return sorted_population[-self.num_elites:], sorted_population[:-self.num_elites]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    random.seed(123)
    n_cities = 30
    tsp = TravellingSalesman.create_random(n_cities)
    mut1 = InversionMutator(0.05)
    mut2 = TranslocationMutator(0.05)
    def mut(x):
        x = mut1(x)
        return mut2(x)
    cx = CycleCrossover()
    sel = ProportionalSelector(30)
    pop = [PathIndividual.create_random(n_cities) for _ in range(100)]
    ga = GeneticAlgorithm(tsp, pop, sel, cx, mut, 3)

    n_gens = 100
    best = []
    gens = []
    fig, ax = plt.subplots()
    ln, = plt.plot([], [], 'r-', animated=True)

    def init():
        ax.set_xlim(0, n_gens)
        ax.set_ylim(0, .0015)
        return ln,

    def update(gen):
        gens.append(gen)
        best.append(ga.best.score)
        ga.evolve()
        ln.set_data(gens, best)
        return ln,

    ani = FuncAnimation(fig, update, frames=n_gens,
                        init_func=init, blit=True)
    plt.show()



    print("best found: ", tsp.best_known.score)  # 0.0011054872608938944
