import random
from abc import ABCMeta, abstractmethod
from typing import Tuple, List

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


class Selection(metaclass=ABCMeta):
    def __init__(self, num_select: int, num_elite: int = 0):
        if num_elite > num_select:
            raise ValueError(f"Number of elites higher than total number selected: {num_elite} > {num_select}")
        self.num_select = num_select - num_elite
        self.num_elite = num_elite

    def _split_elites(self, population: List[Individual]) -> Tuple[List[Individual], List[Individual]]:
        """
        :param population: Population to be split into elites and non-elites
        :return: (elites, non-elites)
        """
        if self.num_elite == 0:
            return [], population
        sorted_population = sorted(population)
        return sorted_population[:self.num_elite], sorted_population[self.num_elite:]

    @abstractmethod
    def __call__(self, population: List[Individual]) -> List[Individual]:
        pass


class ProportionalSelection(Selection):
    def __init__(self, num_select: int, num_elite: int = 0):
        super().__init__(num_select, num_elite)

    def __call__(self, population: List[Individual]) -> List[Individual]:
        if len(population) < self.num_select:
            raise ValueError("Population smaller than num_select")
        elites, non_elites = self._split_elites(population)
        weights = [1/indiv.score for indiv in non_elites]
        return elites + random.choices(non_elites, weights, k=self.num_select)
