import unittest

from pythovolve.mutation import InversionMutator, TranslocationMutator

from pythovolve.individuals import PathIndividual


class _BaseTestMutator(unittest.TestCase):
    individual_class = None
    mutator = None

    def test_create_legal_path(self):
        path_len = 12
        indiv1 = self.individual_class.create_random(path_len)
        indiv2 = self.mutator(indiv1)
        self.assertEqual(list(range(path_len)), sorted(indiv2.phenotype))


class TestInversionMutator(_BaseTestMutator):
    individual_class = PathIndividual
    mutator = InversionMutator(1)


class TestTranslocationMutator(_BaseTestMutator):
    individual_class = PathIndividual
    mutator = TranslocationMutator(1)


# del base tests to prevent execution
del _BaseTestMutator

if __name__ == '__main__':
    unittest.main()
