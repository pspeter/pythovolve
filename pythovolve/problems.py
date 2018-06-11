import random
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from multiprocessing import Process
from pathlib import Path
from multiprocessing import Queue
from typing import List, Tuple, Dict, Union

from sympy.parsing.sympy_parser import parse_expr

from pythovolve.plotting import TSPPlot, ProgressPlot
from .individuals import Individual, PathIndividual, RealValueIndividual

Point = namedtuple("Point", ["x", "y"])
Area = namedtuple("Area", ["min", "max"])


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
    def __init__(self, best_known: Individual = None):
        self.best_known: Individual = None
        if best_known:
            self.score_individual(best_known)

    @abstractmethod
    def score_individual(self, individual: Individual) -> None:
        pass

    @abstractmethod
    def create_individual(self) -> Individual:
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_path: Path, best_known: Individual = None):
        pass

    @abstractmethod
    def get_plot_process(self, max_generations: int) -> Tuple[Queue, Process]:
        pass


class TravellingSalesman(Problem):
    valid_individuals: Dict[str, Individual] = {"path": PathIndividual}

    def __init__(self, cities: List[City], best_known: Individual = None):
        super().__init__(best_known)
        self.cities = cities

    @property
    def defined_area(self) -> Area:
        margin = 0.1

        x_range = min(self.cities, key=lambda c: c.x).x, max(self.cities, key=lambda c: c.x).x
        y_range = min(self.cities, key=lambda c: c.y).y, max(self.cities, key=lambda c: c.y).y

        span_x = x_range[1] - x_range[0]
        span_y = y_range[1] - y_range[0]

        min_point = Point(x_range[0] - span_x * margin, y_range[0] - span_y * margin)
        max_point = Point(x_range[1] + span_x * margin, y_range[1] + span_y * margin)
        area = Area(min_point, max_point)
        return area

    @classmethod
    def create_random(cls, num_cities: int,
                      x_range: Tuple[float, float] = (0, 100),
                      y_range: Tuple[float, float] = (0, 100)) -> "TravellingSalesman":
        if num_cities < 4:
            raise ValueError("Number of cities smaller than 4")
        cities = [City.create_random(x_range, y_range) for _ in range(num_cities)]
        return cls(cities)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], best_known: float = None) -> Problem:
        file_path = Path(file_path)
        cities = []
        with file_path.open() as fp:  # todo improve tsp parsing
            line = fp.readline()
            while line != "NODE_COORD_SECTION\n":
                line = fp.readline()

            line = fp.readline()
            while line != "EOF\n":
                _, x, y = line.split()
                cities.append(City(float(x), float(y)))
                line = fp.readline()

        return cls(cities, best_known)

    def create_individual(self, individual_type: str = "path") -> Individual:
        if individual_type in self.valid_individuals:
            return self.valid_individuals[individual_type].create_random(len(self.cities))
        raise ValueError(f"Type {individual_type} not supported. Expected one of {self.valid_individuals}")

    def score_individual(self, individual: Individual) -> None:
        super().score_individual(individual)

        if isinstance(individual, PathIndividual):
            path = [self.cities[idx] for idx in individual.phenotype]
            score = path[-1].distance(path[0])

            for start, dest in zip(path, path[1:]):
                score += start.distance(dest)

            individual.score = score

            if not self.best_known or individual < self.best_known:
                self.best_known = individual
        else:
            raise ValueError(f"Individuals of type {type(individual).__name__} "
                             f"are not supported by {type(self).__name__}")

    def get_plot_process(self, max_generations: int) -> Tuple[Queue, Process]:
        data_queue = Queue()
        plot_process = Process(target=TSPPlot, args=(max_generations, self, data_queue))

        return data_queue, plot_process

    def __repr__(self):
        return f"{type(self).__name__}(num_cities={len(self.cities)})"


class MultiDimFunction(Problem):
    valid_individuals: Dict[str, Individual] = {"real": RealValueIndividual}

    def __init__(self, expression: str, vector_length: int,
                 search_domain: Tuple[float, float] = (-1, 1),
                 best_known: Individual = None):
        self.expression = expression
        self.vector_length = vector_length
        self.search_domain = search_domain
        super().__init__(best_known)

    @classmethod
    def from_file(cls, file_path: Path, best_known: Individual = None) -> Problem:
        raise NotImplementedError()

    def create_individual(self, individual_type: str = "real") -> Individual:
        if individual_type in self.valid_individuals:
            ind = self.valid_individuals[individual_type]
            return ind.create_random(self.vector_length, self.search_domain)

        raise ValueError(f"Type {individual_type} not supported. Expected one of {self.valid_individuals}")

    def score_individual(self, individual: Individual) -> None:
        if isinstance(individual, RealValueIndividual):
            super().score_individual(individual)

            individual.score = float(parse_expr(self.expression, {"x": individual.phenotype}))

            if not self.best_known or individual < self.best_known:
                self.best_known = individual
        else:
            raise ValueError(f"Individuals of type {type(individual).__name__} "
                             f"are not supported by {type(self).__name__}")

    def get_plot_process(self, max_generations: int) -> Tuple[Queue, Process]:
        data_queue = Queue()
        plot_process = Process(target=ProgressPlot, args=(max_generations, data_queue))

        return data_queue, plot_process

    def __repr__(self):
        return f"{type(self).__name__}(expression={self.expression})"


sphere_problem = MultiDimFunction("x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2", 5, (-5.12, 5.12),
                                  best_known=RealValueIndividual([0, 0, 0, 0, 0], (-5.12, 5.12)))

_goldstein_price = "(1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)) *" \
                   "(30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))"
goldstein_price_problem = MultiDimFunction(_goldstein_price, 2, (-2, 2),
                                           best_known=RealValueIndividual([0, -1], (-2, 2)))

booth_problem = MultiDimFunction("(x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2", 2, (-10, 10),
                                 best_known=RealValueIndividual([1, 3], (-10, 10)))

hoelder_table_problem = MultiDimFunction("-Abs(sin(x[0])*cos(x[1])*exp(1-((x[0]**2+x[1]**2)**0.5)/pi))", 2,
                                         (-10, 10), best_known=RealValueIndividual([0.805502, 9.66459], (-10, 10)))
