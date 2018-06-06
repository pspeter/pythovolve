from pythovolve.algorithm import GeneticAlgorithm
from pythovolve.mutation import inversion_mutator, translocation_mutator


mutators = [inversion_mutator, translocation_mutator]
ga = GeneticAlgorithm()
