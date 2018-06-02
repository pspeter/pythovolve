import random
from abc import ABCMeta, abstractmethod
from typing import Sequence

from pythovolve.individuals import Individual, BinaryIndividual, PathIndividual, RealValueIndividual


class Mutator(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args) -> Individual:
        pass

class BitFlipMutator(Mutator):
    def __init__(self, max_flips: int, probability: float = 0.1):
        self.probability = probability
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


class PathMutator(Mutator, metaclass=ABCMeta):
    def __init__(self, probability: float = 0.1):
        self.probability = probability

    @abstractmethod
    def __call__(self, individual: PathIndividual) -> PathIndividual:
        pass


class InversionMutator(PathMutator):
    def __call__(self, individual: PathIndividual) -> PathIndividual:
        if random.random() > self.probability:
            return individual

        path = individual.phenotype
        start, end = random.sample(list(range(len(path))), 2)
        path[start:end] = reversed(path[start:end])
        return individual


class TranslocationMutator(PathMutator):
    def __call__(self, individual: PathIndividual) -> PathIndividual:
        if random.random() > self.probability:
            return individual

        path = individual.phenotype
        start, mid, end = random.sample(list(range(len(path))), 3)
        path[start:mid] = reversed(path[start:mid])
        path[mid:end] = reversed(path[mid:end])
        return individual


class MultiPathMutator(PathMutator):
    def __init__(self, mutators: Sequence[PathMutator], weights: Sequence[float] = None):
        super().__init__(1)
        self.mutators = mutators
        self.weights = weights

    def __call__(self, individual: PathIndividual) -> PathIndividual:
        mutator = random.choices(self.mutators, self.weights, k=1)[0]
        return mutator(individual)


class RealValueMutator(Mutator):
    def __init__(self, probability: float = 0.1):
        self.probability = probability

    def __call__(self, individual: RealValueIndividual, sigma: float) -> RealValueIndividual:
        if random.random() > self.probability:
            return individual

        vector = individual.phenotype
        min_val, max_val = individual.value_range

        for i in range(len(vector)):
            vector[i] += sigma * random.uniform(-1, 1)
            vector[i] = max(min_val, min(max_val, vector[i]))

        return individual


inversion_mutator = InversionMutator()
translocation_mutator = TranslocationMutator()
multi_path_mutator = MultiPathMutator((inversion_mutator, translocation_mutator))
real_value_mutator = RealValueMutator()
