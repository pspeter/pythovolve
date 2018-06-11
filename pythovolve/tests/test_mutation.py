import unittest

from pythovolve.individuals import PathIndividual
from pythovolve.mutation import InversionMutator, TranslocationMutator


class TestInversionMutator(unittest.TestCase):

    def setUp(self):
        self.mutator = InversionMutator(1)

    def test_create_legal_path(self):
        path_len = 12
        indiv1 = PathIndividual.create_random(path_len)
        indiv2 = self.mutator(indiv1)
        self.assertEqual(list(range(path_len)), sorted(indiv2.phenotype))


class TestTranslocationMutator(unittest.TestCase):

    def setUp(self):
        self.mutator = TranslocationMutator(1)

    def test_create_legal_path(self):
        path_len = 12
        indiv1 = PathIndividual.create_random(path_len)
        indiv2 = self.mutator(indiv1)
        self.assertEqual(list(range(path_len)), sorted(indiv2.phenotype))


if __name__ == '__main__':
    unittest.main()
