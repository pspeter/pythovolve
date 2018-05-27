import random
from abc import ABCMeta, abstractmethod
from typing import Tuple

from .individuals import Individual, BinaryIndividual, PathIndividual


class Mutator(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, individual: Individual):
        pass


class BitFlipMutator(Mutator):
    def __init__(self, max_flips: int):
        self.max_flips = max_flips

    def __call__(self, individual: BinaryIndividual):
        len_pheno = len(individual.phenotype)
        num_flips = random.randint(1, max(len_pheno, self.max_flips))
        to_flip = random.sample(range(len_pheno), num_flips)
        for flip_index in to_flip:
            individual.phenotype[flip_index] ^= True  # flips bit
        return individual


class InversionMutator(Mutator):
    def __call__(self, individual: PathIndividual):
        path = individual.phenotype
        start, end = random.sample(list(range(len(path))), 2)
        path[start:end] = reversed(path[start:end])
        return individual


class TranslocationMutator(Mutator):
    def __call__(self, individual: PathIndividual):
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
