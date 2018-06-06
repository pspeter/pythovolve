from pythovolve.individuals import PathIndividual, RealValueIndividual
from pythovolve.crossover import CycleCrossover, OrderCrossover, SinglePointCrossover
import unittest


class _BaseTestCrossover(unittest.TestCase):
    individual_class = None
    crossover = None

    def test_children_not_always_equal(self):
        path_len = 12
        for _ in range(10):
            indiv1 = self.individual_class.create_random(path_len)
            indiv2 = self.individual_class.create_random(path_len)
            indiv3, indiv4 = self.crossover(indiv1, indiv2)
            if indiv1.phenotype != indiv3.phenotype and \
                    indiv2.phenotype != indiv3.phenotype and \
                    indiv1.phenotype != indiv4.phenotype and \
                    indiv2.phenotype != indiv4.phenotype:
                break
        else:
            self.assertTrue(False)  # crossover never created different children


class _BaseTestPathCrossover(_BaseTestCrossover):
    individual_class = PathIndividual

    def test_creates_legal_path(self):
        path_len = 12
        indiv1 = self.individual_class.create_random(path_len)
        indiv2 = self.individual_class.create_random(path_len)
        indiv3, indiv4 = self.crossover(indiv1, indiv2)
        self.assertEqual(list(range(path_len)), sorted(indiv3.phenotype))
        self.assertEqual(list(range(path_len)), sorted(indiv4.phenotype))


class TestCycleCrossover(_BaseTestPathCrossover):
    crossover = CycleCrossover()


class TestOrderCrossover(_BaseTestPathCrossover):
    crossover = OrderCrossover()


class TestSinglePointCrossover(_BaseTestCrossover):
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


# del base tests to prevent execution
del _BaseTestCrossover
del _BaseTestPathCrossover

if __name__ == '__main__':
    unittest.main()
