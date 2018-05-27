import random
from typing import Any, List, Callable


class Individual:
    def __init__(self, phenotype: Any):
        self._score: float = None
        self.phenotype = phenotype

    @property
    def score(self) -> float:
        if not self._score:
            raise ValueError("Individual has not been scored yet")
        return self._score

    def calculate_score(self, score_func: Callable):
        self._score = score_func(self)

    def __repr__(self):
        return f"{type(self).__name__}({self.phenotype})"


class BinaryIndividual(Individual):

    def __init__(self, phenotype: List[bool]):
        super().__init__(phenotype)

    @classmethod
    def create_random(cls, num_bits: int):
        return cls([random.choice([True, False] for _ in range(num_bits))])

    def __str__(self):
        return "".join(("1" if bit else "0" for bit in self.phenotype))


class PathIndividual(Individual):
    phenotype: List[int] = None

    def __init__(self, phenotype: List[int]):
        super().__init__(phenotype)

    @classmethod
    def create_random(cls, num_cities: int):
        return cls(random.sample(range(num_cities), num_cities))

    def __str__(self):
        return "[" + " -> ".join((str(city) for city in self.phenotype)) + "]"
