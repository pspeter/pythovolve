import random
from abc import ABCMeta, abstractmethod
from typing import Sequence

from pythovolve.individuals import Individual


class Selector(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, population: Sequence[Individual]) -> Individual:
        pass


class ProportionalSelector(Selector):
    def __init__(self):
        super().__init__()

    def __call__(self, population: Sequence[Individual]) -> Individual:
        weights = [-indiv.score for indiv in population]
        weights = [w - min(weights) + 1 for w in weights]  # shift to positive
        return random.choices(population, weights, k=1)[0]


class LinearRankSelector(Selector):
    def __init__(self):
        super().__init__()

    def __call__(self, population: Sequence[Individual]) -> Individual:
        population = sorted(population, reverse=True)
        ranks = range(1, len(population) + 1)
        sum_ranks = sum(ranks)
        weights = [i / sum_ranks for i in ranks]
        return random.choices(population, weights, k=1)[0]


class TournamentSelector(Selector):
    def __init__(self):
        super().__init__()

    def __call__(self, population: Sequence[Individual], tournament_size=5) -> Individual:
        chosen = random.choices(population, k=tournament_size)
        return min(chosen, key=lambda ind: ind.score)


proportional_selector = ProportionalSelector()
linear_rank_selector = LinearRankSelector()
tournament_selector = TournamentSelector()