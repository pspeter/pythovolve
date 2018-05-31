import random
from typing import Tuple, List

from matplotlib.animation import FuncAnimation

from pythovolve.problems import Problem, TravellingSalesman
from pythovolve.individuals import Individual, PathIndividual
from pythovolve.crossover import CycleCrossover, Crossover
from pythovolve.selection import Selector, ProportionalSelector
from pythovolve.mutation import Mutator, TranslocationMutator, InversionMutator


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
