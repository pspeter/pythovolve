import random
from functools import total_ordering
from typing import Any, List, Callable, Tuple


@total_ordering
class Individual:
    """Base class for individuals."""

    def __init__(self, phenotype: Any):
        self._score: float = None
        self.phenotype = phenotype

    @property
    def score(self) -> float:
        """*Smaller* score is always *better*. Score should
            not be smaller or equal to 0."""
        if not self._score:
            raise ValueError("Individual has not been scored yet")
        return self._score

    @score.setter
    def score(self, score: float) -> None:
        self._score = score

    @classmethod
    def create_random(cls, *args):
        raise NotImplementedError(f"Derived class '{cls}' has not implemented this method")

    def __repr__(self):
        return f"{type(self).__name__}({self.phenotype})"

    def __lt__(self, other):
        if not isinstance(other, Individual):
            return NotImplemented
        try:
            return self.score < other.score
        except ValueError:
            return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, Individual):
            return NotImplemented
        try:
            return self.score == other.score
        except ValueError:
            return NotImplemented


class BinaryIndividual(Individual):
    def __init__(self, phenotype: List[bool]):
        super().__init__(phenotype)

    @classmethod
    def create_random(cls, size: int):
        return cls([random.choice([True, False] for _ in range(size))])

    def __str__(self):
        return "".join(("1" if bit else "0" for bit in self.phenotype))


class PathIndividual(Individual):
    def __init__(self, phenotype: List[int]):
        super().__init__(phenotype)

    @classmethod
    def create_random(cls, size: int):
        return cls(random.sample(range(size), size))

    def __str__(self):
        return "[" + " -> ".join((str(city) for city in self.phenotype)) + "]"


class RealValueIndividual(Individual):
    def __init__(self, phenotype: List[float], value_range: Tuple[float, float] = (-1, 1)):
        super().__init__(phenotype)
        self.value_range = value_range

    @classmethod
    def create_random(cls, size: int, value_range: Tuple[float, float] = (-1, 1)):
        return cls([random.uniform(*value_range) for _ in range(size)])

    def __str__(self):
        return "[" + " -> ".join((f"{value:.3f}" for value in self.phenotype)) + "]"
