import random
from abc import abstractmethod, ABCMeta
from typing import Any, List


class Individual(metaclass=ABCMeta):
    @property
    @abstractmethod
    def phenotype(self) -> Any:
        pass

    @phenotype.setter
    @abstractmethod
    def phenotype(self, phenotype: Any) -> None:
        pass

    def __repr__(self):
        return f"{type(self).__name__}({self.phenotype})"


class BinaryIndividual(Individual):
    phenotype: List[bool] = None

    def __init__(self, binary: List[bool]):
        super().__init__()
        self.phenotype = binary

    @classmethod
    def create_random(cls, num_bits: int):
        return cls([random.choice([True, False] for _ in range(num_bits))])

    def __str__(self):
        return "".join(("1" if bit else "0" for bit in self.phenotype))


class PathIndividual(Individual):
    phenotype: List[int] = None

    def __init__(self, path: List[int]):
        super().__init__()
        self.phenotype = path

    @classmethod
    def create_random(cls, num_cities: int):
        return cls(random.sample(range(num_cities), num_cities))

    def __str__(self):
        return "[" + " -> ".join((str(city) for city in self.phenotype)) + "]"


class Population:
    pass


