from unittest.mock import MagicMock, patch

from pythovolve.individuals import PathIndividual, RealValueIndividual
from pythovolve.crossover import CycleCrossover, OrderCrossover, SinglePointCrossover
import unittest


class TestCycleCrossover(unittest.TestCase):

    def setUp(self):
        self.crossover = CycleCrossover()

    def test_creates_legal_path(self):
        path_len = 12
        indiv1 = PathIndividual.create_random(path_len)
        indiv2 = PathIndividual.create_random(path_len)
        indiv3, indiv4 = self.crossover(indiv1, indiv2)

        self.assertEqual(list(range(path_len)), sorted(indiv3.phenotype))
        self.assertEqual(list(range(path_len)), sorted(indiv4.phenotype))

    def test_children_not_always_equal(self):
        path_len = 12
        for _ in range(50):
            indiv1 = PathIndividual.create_random(path_len)
            indiv2 = PathIndividual.create_random(path_len)
            indiv3, indiv4 = self.crossover(indiv1, indiv2)

            if indiv1.phenotype != indiv3.phenotype and \
                    indiv2.phenotype != indiv3.phenotype and \
                    indiv1.phenotype != indiv4.phenotype and \
                    indiv2.phenotype != indiv4.phenotype:
                break
        else:
            self.assertTrue(False)  # crossover never created different children

    def test_crossover_call(self):

        indiv1 = PathIndividual([4, 1, 0, 3, 2])
        indiv2 = PathIndividual([1, 4, 0, 2, 3])

        with patch("random.randrange", lambda _: 3):
            child1, child2 = self.crossover(indiv1, indiv2)

        self.assertEqual(child1.phenotype, [1, 4, 0, 3, 2])
        self.assertEqual(child2.phenotype, [4, 1, 0, 2, 3])


class TestOrderCrossover(unittest.TestCase):

    def setUp(self):
        self.crossover = OrderCrossover()

    def test_creates_legal_path(self):
        path_len = 12
        indiv1 = PathIndividual.create_random(path_len)
        indiv2 = PathIndividual.create_random(path_len)
        indiv3, indiv4 = self.crossover(indiv1, indiv2)
        self.assertEqual(list(range(path_len)), sorted(indiv3.phenotype))
        self.assertEqual(list(range(path_len)), sorted(indiv4.phenotype))

    def test_children_not_always_equal(self):
        path_len = 12
        for _ in range(10):
            indiv1 = PathIndividual.create_random(path_len)
            indiv2 = PathIndividual.create_random(path_len)
            indiv3, indiv4 = self.crossover(indiv1, indiv2)
            if indiv1.phenotype != indiv3.phenotype and \
                    indiv2.phenotype != indiv3.phenotype and \
                    indiv1.phenotype != indiv4.phenotype and \
                    indiv2.phenotype != indiv4.phenotype:
                break
        else:
            self.assertTrue(False)  # crossover never created different children


class TestSinglePointCrossover(unittest.TestCase):
    individual_class = RealValueIndividual
    crossover = SinglePointCrossover()

    def test_children_keep_same_total_sum(self):
        indiv1 = RealValueIndividual.create_random(20, (0, 10))
        indiv2 = RealValueIndividual.create_random(20, (0, 10))

        crossover = SinglePointCrossover()
        indiv3, indiv4 = crossover(indiv1, indiv2)

        children_sum = sum(indiv1.phenotype) + sum(indiv2.phenotype)
        parents_sum = sum(indiv3.phenotype) + sum(indiv4.phenotype)

        self.assertAlmostEqual(children_sum, parents_sum)

    def test_children_not_always_equal(self):
        path_len = 12
        for _ in range(10):
            indiv1 = RealValueIndividual.create_random(path_len)
            indiv2 = RealValueIndividual.create_random(path_len)
            indiv3, indiv4 = self.crossover(indiv1, indiv2)
            if indiv1.phenotype != indiv3.phenotype and \
                    indiv2.phenotype != indiv3.phenotype and \
                    indiv1.phenotype != indiv4.phenotype and \
                    indiv2.phenotype != indiv4.phenotype:
                break
        else:
            self.assertTrue(False)  # crossover never created different children


if __name__ == '__main__':
    unittest.main()
