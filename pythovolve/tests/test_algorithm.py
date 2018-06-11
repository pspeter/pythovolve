import unittest

from pythovolve.individuals import PathIndividual
from pythovolve.problems import TravellingSalesman

from pythovolve.algorithm import GeneticAlgorithm, EvolutionAlgorithm
from pythovolve.selection import ProportionalSelector

from pythovolve.crossover import CycleCrossover
from pythovolve.mutation import InversionMutator, TranslocationMutator


class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        self.problem = TravellingSalesman.create_random(20)
        self.mutators = [InversionMutator(1), TranslocationMutator(1)]
        self.crossover = CycleCrossover()
        self.selectors = ProportionalSelector()

    def test_init(self):
        population_size = 25
        ga = GeneticAlgorithm(self.problem, self.selectors, self.crossover,
                              self.mutators, population_size=population_size,
                              num_elites=10, max_generations=10, callbacks=None,
                              plot_progress=False, verbosity=0)

        self.assertIsInstance(ga, EvolutionAlgorithm)
        self.assertIsInstance(ga, GeneticAlgorithm)

        self.assertEqual(len(ga.population), population_size)
        self.assertIsInstance(ga.best, PathIndividual)
        self.assertIsInstance(ga.best.score, float)
        self.assertEqual(ga.generation, 0)

    def test_evolve_once(self):
        population_size = 25
        ga = GeneticAlgorithm(self.problem, self.selectors, self.crossover,
                              self.mutators, population_size=population_size,
                              num_elites=10, max_generations=10, callbacks=None,
                              plot_progress=False, verbosity=0)

        first_best = ga.best
        ga.evolve_once()

        self.assertEqual(len(ga.population), population_size)
        self.assertIsInstance(ga.best, PathIndividual)
        self.assertTrue(first_best.score > ga.best.score or first_best is ga.best)
        self.assertEqual(ga.generation, 1)

        ga.evolve_once()
        self.assertEqual(len(ga.population), population_size)
        self.assertTrue(first_best.score > ga.best.score or first_best is ga.best)
        self.assertIsInstance(ga.best, PathIndividual)
        self.assertEqual(ga.generation, 2)

    def test_evolve(self):
        population_size = 25
        max_generations = 10
        ga = GeneticAlgorithm(self.problem, self.selectors, self.crossover,
                              self.mutators, population_size=population_size,
                              num_elites=10, max_generations=10, callbacks=None,
                              plot_progress=False, verbosity=0)

        first_best = ga.best
        ga.evolve()

        self.assertEqual(len(ga.population), population_size)
        self.assertIsInstance(ga.best, PathIndividual)
        self.assertTrue(first_best.score > ga.best.score or first_best is ga.best)
        self.assertEqual(ga.generation, max_generations)

