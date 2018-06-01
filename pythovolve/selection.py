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
        weights = [indiv.score for indiv in population]
        return random.choices(population, weights, k=1)[0]


class LinearRankSelector(Selector):
    def __init__(self):
        super().__init__()

    def __call__(self, population: Sequence[Individual]) -> Individual:
        population = sorted(population, key=lambda ind: ind.score)
        ranks = range(1, len(population) + 1)
        sum_ranks = sum(ranks)
        weights = [i / sum_ranks for i in ranks]
        return random.choices(population, weights, k=1)[0]


class TournamentSelector(Selector):
    def __init__(self):
        super().__init__()

    def __call__(self, population: Sequence[Individual], tournament_size=5) -> Individual:
        chosen = random.choices(population, k=tournament_size)
        return max(chosen, key=lambda ind: ind.score)


class MultiSelector(Selector):
    def __init__(self, selectors: Sequence[Selector], weights: Sequence[float] = None):
        super().__init__()
        self.selectors = selectors
        self.weights = weights

    def __call__(self, population: Sequence[Individual]) -> Individual:
        selector = random.choices(self.selectors, self.weights, k=1)[0]
        return selector(population)


proportional_selector = ProportionalSelector()
linear_rank_selector = LinearRankSelector()
tournament_selector = TournamentSelector()
multi_selector = MultiSelector((proportional_selector, linear_rank_selector, tournament_selector))
