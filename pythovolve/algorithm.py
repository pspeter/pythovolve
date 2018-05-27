from abc import ABCMeta, abstractmethod
import random
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

