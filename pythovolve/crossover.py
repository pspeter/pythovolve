import random
from abc import ABCMeta, abstractmethod
from typing import Tuple

from pythovolve.individuals import Individual, PathIndividual, RealValueIndividual


class Crossover(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, mother: Individual, father: Individual) -> Tuple[Individual, Individual]:
        pass


class CycleCrossover(Crossover):
    def __call__(self, mother: PathIndividual, father: PathIndividual) \
            -> Tuple[PathIndividual, PathIndividual]:
        first = random.randrange(len(mother.phenotype))
        return self._create_child(mother, father, first), self._create_child(father, mother, first)

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


class OrderCrossover(Crossover):
    def __call__(self, mother: PathIndividual, father: PathIndividual) \
            -> Tuple[PathIndividual, PathIndividual]:
        idx1, idx2 = random.sample(range(len(mother.phenotype)), k=2)
        return self._create_child(mother, father, idx1, idx2), self._create_child(father, mother, idx1, idx2)

    @staticmethod
    def _create_child(parent1: PathIndividual, parent2: PathIndividual,
                      start: int, end: int) -> PathIndividual:
        child = parent1.phenotype[:]
        if start > end:
            included = child[:end] + child[start:]
        else:
            included = child[start:end]

        missing = [g for g in parent2.phenotype if g not in included]

        if start > end:
            child[end:start] = missing
        else:
            child[:start] = missing[:start]
            child[end:] = missing[start:]

        return PathIndividual(child)


class SinglePointCrossover(Crossover):
    def __call__(self, mother: RealValueIndividual, father: RealValueIndividual) \
            -> Tuple[RealValueIndividual, RealValueIndividual]:
        start, = random.sample(range(len(mother.phenotype)), k=1)
        child1 = mother.phenotype[:start] + father.phenotype[start:]
        child2 = father.phenotype[:start] + mother.phenotype[start:]
        return RealValueIndividual(child1, mother.value_range), RealValueIndividual(child2, mother.value_range)


cycle_crossover = CycleCrossover()
order_crossover = OrderCrossover()
single_point_crossover = SinglePointCrossover()
