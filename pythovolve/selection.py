import random
from abc import ABCMeta, abstractmethod
from typing import List

from pythovolve.individuals import Individual


class Selector(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, population: List[Individual]) -> Individual:
        pass


class ProportionalSelector(Selector):
    def __init__(self):
        super().__init__()

    def __call__(self, population: List[Individual]) -> Individual:
        weights = [indiv.score for indiv in population]
        return random.choices(population, weights, k=1)[0]


class LinearRankSelector(Selector):
    def __init__(self):
        super().__init__()

    def __call__(self, population: List[Individual]) -> Individual:
        population = sorted(population, key=lambda ind: ind.score)
        ranks = range(1, len(population) + 1)
        sum_ranks = sum(ranks)
        weights = [i / sum_ranks for i in ranks]
        return random.choices(population, weights, k=1)[0]


class TournamentSelector(Selector):
    def __init__(self):
        super().__init__()

    def __call__(self, population: List[Individual], tournament_size=5) -> Individual:
        chosen = random.choices(population, k=tournament_size)
        return max(chosen, key=lambda ind: ind.score)

