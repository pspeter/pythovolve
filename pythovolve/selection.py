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
