import random
from abc import ABCMeta, abstractmethod
from typing import Sequence

from pythovolve.individuals import Individual, BinaryIndividual, PathIndividual, RealValueIndividual


class Mutator(metaclass=ABCMeta):
    def __init__(self, probability: float = 0.1):
        self.probability = probability

    @abstractmethod
    def __call__(self, *args) -> Individual:
        pass


class PathMutator(Mutator, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, individual: PathIndividual, *_) -> PathIndividual:
        pass


class BitFlipMutator(Mutator):
    def __init__(self, max_flips: int, probability: float = 0.1):
        super().__init__(probability)
        self.max_flips = max_flips

    def __call__(self, individual: BinaryIndividual, *_) -> BinaryIndividual:
        if random.random() > self.probability:
            return individual

        len_pheno = len(individual.phenotype)
        num_flips = random.randint(1, min(len_pheno, self.max_flips))
        to_flip = random.sample(range(len_pheno), num_flips)

        for flip_index in to_flip:
            individual.phenotype[flip_index] ^= True  # flips bit

        return individual


class InversionMutator(PathMutator):
    def __call__(self, individual: PathIndividual, *_) -> PathIndividual:
        if random.random() > self.probability:
            return individual

        path = individual.phenotype
        start, end = random.sample(list(range(len(path))), 2)
        path[start:end] = reversed(path[start:end])
        return individual


class TranslocationMutator(PathMutator):
    def __call__(self, individual: PathIndividual, *_) -> PathIndividual:
        if random.random() > self.probability:
            return individual

        path = individual.phenotype
        start, mid, end = random.sample(list(range(len(path))), 3)
        path[start:mid] = reversed(path[start:mid])
        path[mid:end] = reversed(path[mid:end])
        return individual


class SigmaMutator(Mutator, metaclass=ABCMeta):
    def __init__(self, inital_sigma: float, sigma_multiplier: float = 1.15, probability: float = 1):
        super().__init__(probability)
        self.sigma_multiplier = sigma_multiplier
        self.sigma = inital_sigma

    @abstractmethod
    def __call__(self, individual: Individual):
        pass

    def adapt_sigma(self, percent_success: float):
        """Sigma adaption as suggested by Schwefel (1981)"""
        if percent_success > 1 / 5:
            self.sigma *= self.sigma_multiplier
        else:
            self.sigma /= self.sigma_multiplier
        print(self.sigma)


class GaussNoiseMutator(SigmaMutator):
    def __init__(self, probability: float = 1):
        super().__init__(probability)

    def __call__(self, individual: RealValueIndividual, sigma: float = None) -> RealValueIndividual:
        if random.random() > self.probability:
            return individual

        vector = individual.phenotype
        min_val, max_val = individual.value_range

        if sigma is None:
            sigma = (max_val - min_val) / 5  # GA uses 1/5 of range as sigma todo better way?

        for i in range(len(vector)):
            vector[i] += random.gauss(0, sigma)
            vector[i] = max(min_val, min(max_val, vector[i]))

        return individual


inversion_mutator = InversionMutator()
translocation_mutator = TranslocationMutator()
real_value_mutator = GaussNoiseMutator()
