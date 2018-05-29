import random
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from .individuals import Individual, PathIndividual


class City:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance(self, destination) -> float:
        return (abs(self.x - destination.x) ** 2 + abs(self.y - destination.y) ** 2) ** 0.5

    @staticmethod
    def create_random(x_range: Tuple[float, float], y_range: Tuple[float, float]):
        return City(random.uniform(*x_range), random.uniform(*y_range))


class Problem(metaclass=ABCMeta):
    @abstractmethod
    def score_individual(self, individual: Individual) -> float:
        pass


class TravellingSalesman(Problem):
    def __init__(self, cities: List[City], best_known: Individual = None):
        self.cities = cities
        self.best_known = best_known

    @classmethod
    def create_random(cls, num_cities,
                      x_range: Tuple[float, float] = (0, 100),
                      y_range: Tuple[float, float] = (0, 100)):
        cities = [City.create_random(x_range, y_range) for _ in range(num_cities)]
        return cls(cities)

    def score_individual(self, individual: Individual) -> None:
        super().score_individual(individual)

        if isinstance(individual, PathIndividual):
            path = [self.cities[idx] for idx in individual.phenotype]
            total_distance = 0.0

            for start, dest in zip(path, path[1:]):
                total_distance += start.distance(dest)

            total_distance += path[-1].distance(path[0])
            score = 1 / total_distance
            individual.score = score

            if not self.best_known or individual > self.best_known:
                self.best_known = individual
        else:
            raise ValueError(f"Individuals of type {type(individual)} are not supported by {type(TravellingSalesman)}")

    def __repr__(self):
        return f"{type(self).__name__}(num_cities={len(self.cities)})"
