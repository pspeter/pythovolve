import random
from abc import ABCMeta, abstractmethod
from typing import Sequence

from pythovolve.individuals import Individual, BinaryIndividual, PathIndividual


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


class MultiMutator(Mutator):
    def __init__(self, mutators: Sequence[Mutator], weights: Sequence[float] = None):
        super().__init__()
        self.mutators = mutators
        self.weights = weights

    def __call__(self, population: Sequence[Individual]) -> Individual:
        mutator = random.choices(self.mutators, self.weights, k=1)[0]
        return mutator(population)


inversion_mutator = InversionMutator()
translocation_mutator = TranslocationMutator()
multi_mutator = MultiMutator((inversion_mutator, translocation_mutator))
