import random
import unittest

from pythovolve.individuals import  PathIndividual

from pythovolve.selection import ProportionalSelector, TournamentSelector, LinearRankSelector


class TestProportionalSelector(unittest.TestCase):

    def setUp(self):
        self.selector = ProportionalSelector()

    def test_call_returns_single_individual(self):
        individual_class = PathIndividual
        population = [individual_class.create_random(5) for _ in range(10)]

        for individual in population:
            individual.score = random.random()

        single = self.selector(population)
        self.assertIsInstance(single, individual_class)


class TestTournamentSelector(unittest.TestCase):
    individual_class = PathIndividual

    def setUp(self):
        self.selector = TournamentSelector()

    def test_call_returns_single_individual(self):
        individual_class = PathIndividual
        population = [individual_class.create_random(5) for _ in range(10)]

        for individual in population:
            individual.score = random.random()

        single = self.selector(population)
        self.assertIsInstance(single, individual_class)


class TestLinearRankSelector(unittest.TestCase):
    individual_class = PathIndividual

    def setUp(self):
        self.selector = LinearRankSelector()

    def test_call_returns_single_individual(self):
        individual_class = PathIndividual
        population = [individual_class.create_random(5) for _ in range(10)]

        for individual in population:
            individual.score = random.random()

        single = self.selector(population)
        self.assertIsInstance(single, individual_class)


if __name__ == '__main__':
    unittest.main()
