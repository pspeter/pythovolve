from pythovolve.individuals import PathIndividual
from pythovolve.algorithm import CycleCrossover
import unittest


class TestCycleCrossover(unittest.TestCase):

    def test_creates_legal_path(self):
        path_len = 12
        cycle = CycleCrossover()
        indiv1 = PathIndividual.create_random(path_len)
        indiv2 = PathIndividual.create_random(path_len)
        indiv3, indiv4 = cycle((indiv1, indiv2))
        self.assertEqual(list(range(path_len)), sorted(indiv3.phenotype))
        self.assertEqual(list(range(path_len)), sorted(indiv4.phenotype))


if __name__ == '__main__':
    unittest.main()
