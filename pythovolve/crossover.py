import random
from abc import ABCMeta, abstractmethod
from typing import Tuple

from pythovolve.individuals import Individual, PathIndividual


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
