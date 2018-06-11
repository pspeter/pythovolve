import unittest

from pythovolve.individuals import PathIndividual, RealValueIndividual


class TestPathIndividuals(unittest.TestCase):

    def test_random_creation(self):
        size = 100
        individual = PathIndividual.create_random(size)

        self.assertEqual(len(individual.phenotype), size)
        self.assertListEqual(sorted(individual.phenotype), list(range(size)))

    def test_init(self):
        path = [5, 2, 6, 3, 1, 0, 4]
        individual = PathIndividual(path)
        self.assertListEqual(path, individual.phenotype)

    def test_clone(self):
        individual = PathIndividual([5, 2, 6, 3, 1, 0, 4])
        cloned = individual.clone()
        self.assertListEqual(individual.phenotype, cloned.phenotype)


class TestRealValueIndividuals(unittest.TestCase):

    def _individual_is_legal(self, indiv):
        min_val, max_val = indiv.value_range
        for real_value in indiv.phenotype:
            self.assertTrue(min_val <= real_value <= max_val)

    def test_random_creation(self):
        size = 100
        individual = RealValueIndividual.create_random(size)

        self._individual_is_legal(individual)

    def test_init(self):
        values = [0.2, -0.2, 0, -16e125, 73e-125]
        individual = RealValueIndividual(values, (-17e126, 1))

        self._individual_is_legal(individual)

        with self.assertRaises(ValueError):
            RealValueIndividual([], (0, 1))

        with self.assertRaises(ValueError):
            RealValueIndividual([100, 0], (-1, 1))

        with self.assertRaises(ValueError):
            RealValueIndividual([-100, 0], (-1, 1))

    def test_clone(self):
        individual = RealValueIndividual([0.1, 1/32, -0.6, 0., -1e-53, 1e-62], (-1, 1))
        cloned = individual.clone()
        self.assertListEqual(individual.phenotype, cloned.phenotype)
