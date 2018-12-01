import argparse
import warnings
from pathlib import Path
from statistics import mean, stdev

from pythovolve.algorithm import EvolutionStrategy, GeneticAlgorithm, OSGeneticAlgorithm, EvolutionAlgorithm
from pythovolve.callbacks import EarlyStopCallback, TimerCallback, ProgressLoggerCallback
from pythovolve.crossover import CycleCrossover, OrderCrossover, SinglePointCrossover
from pythovolve.mutation import InversionMutator, TranslocationMutator, GaussNoiseMutator
from pythovolve.problems import sphere_problem, goldstein_price_problem, booth_problem, hoelder_table_problem, \
    MultiDimFunction, TravellingSalesman
from pythovolve.selection import LinearRankSelector, ProportionalSelector, TournamentSelector


def _algorithm_from_args(args) -> EvolutionAlgorithm:

    predefined_dict = {
        "sphere": sphere_problem,
        "goldstein_price": goldstein_price_problem,
        "booth": booth_problem,
        "hoelder_table": hoelder_table_problem
    }

    problem_dict = {
        "TSP": TravellingSalesman,
        "MDTF": MultiDimFunction
    }

    algorithm_dict = {
        "ES": EvolutionStrategy,
        "GA": GeneticAlgorithm,
        "OSGA": OSGeneticAlgorithm
    }

    selector_dict = {
        "proportional": ProportionalSelector,
        "linear_rank": LinearRankSelector,
        "tournament": TournamentSelector
    }

    mutator_dict = {
        "inversion": InversionMutator,
        "translocation": TranslocationMutator,
        "gauss": GaussNoiseMutator
    }

    crossover_dict = {
        "cycle": CycleCrossover,
        "order": OrderCrossover,
        "single_point": SinglePointCrossover
    }

    if args.predefined:
        problem = predefined_dict[args.predefined]
    else:
        if args.random is not None:
            problem = problem_dict[args.problem_type].create_random(args.random)
        elif args.problem_file:
            problem = problem_dict[args.problem_type].from_file(Path(args.problem_file))
        else:
            warnings.warn("Cannot create problem with given parameters. Use either "
                          "-d to specify a predefined problem, or use one of -r "
                          "or -f to specify a way to construct a problem.")
            warnings.warn("Defaulting to random TSP with 50 cities.")
            problem = TravellingSalesman.create_random(50)

    Algorithm = algorithm_dict[args.algorithm]

    def handle_multiple_names(name_dict, arg, **constructor_args):
        return [name_dict.get(name)(**constructor_args) for name in arg]

    selectors = handle_multiple_names(selector_dict, args.selectors)
    crossovers = handle_multiple_names(crossover_dict, args.crossovers)
    mutators = handle_multiple_names(mutator_dict, args.mutators, probability=args.mutation_rate)

    kwargs = vars(args).copy()  # copy to not mess with original args for multiple runs
    kwargs["selectors"] = selectors
    kwargs["crossovers"] = crossovers
    kwargs["mutators"] = mutators
    kwargs["callbacks"] = []

    if args.min_sigma or args.no_progress or args.max_seconds:
        kwargs["callbacks"].append(EarlyStopCallback(args.min_sigma, args.no_progress, args.max_seconds))

    if args.verbosity >= 1:
        kwargs["callbacks"].append(TimerCallback())

    if args.verbosity >= 2:
        kwargs["callbacks"].append(ProgressLoggerCallback())

    return Algorithm(problem, **kwargs)


def main():
    description = """Evolutionary Algorithms (EA) are population based optimization 
    algorithms. This framework provides a couple implemented algorithms as well as
     problems typically solved with this group of algorithms. Also, it defines base 
     classes and interfaces that should make it easy to expand on its functionality. 
    Finally, it also provides live visualisation of the optimization process."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("algorithm", type=str, choices=["ES", "GA", "OSGA"],
                        help="Choose which algorithm to use. GA = Genetic Algorithm, "
                             "OSGA = Genetic Algorithm with Offspring "
                             "Selection, ES = Evolution Strategy")

    parser.add_argument("-t", "--problem-type", type=str, choices=["TSP", "MDTF"],
                        default="TSP",
                        help="Choose which problem to solve. TSP = Traveling Salesman"
                             " Problem, MDTF = Multi-Dimensional Test Function.")
    parser.add_argument("-d", "--predefined", choices=["sphere", "goldstein_price",
                                                       "booth", "hoelder_table"],
                        help="Choose a concrete problem to solve. These problems "
                             "contain the best possible solution. If this argument "
                             "is set, -t and -r will be ignored. 'sphere', "
                             "'goldstein_price', 'booth' and 'hoelder_table' are "
                             "MDTF problems.")

    parser.add_argument("-r", "--random", type=int,
                        help="Creates a random problem of the given dimensionality "
                             "and of type -t. For TSP, this argument defines the number "
                             "of cities. Currently this is only supported for "
                             "TSP.")

    parser.add_argument("-f", "--problem-file", type=str,
                        help="Path to file that the problem given by -t can use to "
                             "construct an instance of that problem. Currently only "
                             "supported for TSP using a TSPLIB file that includes "
                             "a NODE_COORD_SECTION, for example 'ch130'.")

    parser.add_argument("-n", "--max-generations", type=int, default=200,
                        help="Maximum amount of generations. Algorithm will stop "
                             "after generating this amount of generations. It might "
                             "stop earlier than that (e.g. due to callbacks).")

    parser.add_argument("-e", "--num-elites", type=int, default=1,
                        help="The top -e individuals of each generation are kept "
                             "without any modification for the next generation. "
                             "(default 1)")

    parser.add_argument("-M", "--population-size", type=int, default=100,
                        help="Size of population (mu). Used by all algorithms. "
                             "(default 100)")

    parser.add_argument("-L", "--num-children", type=int, default=10,
                        help="Number of children (lambda). Only ES uses this value."
                             "(default 10)")

    parser.add_argument("-P", "--keep-parents", action="store_true",
                        help="Decides whether ES selects from the parents and their"
                             "children (True) or only the children (False). Setting "
                             "the -P flag corresponds to mu+lambda, while not setting "
                             "it corresponds to a mu,lambda strategy.")

    parser.add_argument("--max-selection-pressure", type=int, default=50,
                        help="Max selection pressure parameter. Only OSGA uses this"
                             "value. (default 50)")

    parser.add_argument("-m", "--mutators", choices=["inversion", "translocation", "gauss"],
                        nargs="*", default=["inversion", "translocation"],
                        help="Choose which mutator to use. TSP only supports inversion "
                             "and translocation, while MDTF only supports gauss"
                             "currently. You can also specify more than one mutator.")

    parser.add_argument("-R", "--mutation-rate", type=float, default=0.2,
                        help="Probability of mutating each child. (default 0.2)")

    parser.add_argument("-c", "--crossovers", choices=["cycle", "order", "single_point"],
                        nargs="*", default=["cycle", "order"],
                        help="Choose which crossover to use. Note that ES does not use "
                             "crossover. TSP only supports cycle and order, while MDTF "
                             "only supports single_point currently. You can specify more "
                             "than one crossover.")

    parser.add_argument("-s", "--selectors", nargs="*",
                        choices=["proportional", "linear_rank", "tournament"],
                        default=["proportional", "linear_rank", "tournament"],
                        help="Choose which selector to use. You can specify more than "
                             "one. Each call will randomly choose among one of them.")

    parser.add_argument("--metrics", type=str, nargs="*", default=["best", "current_best"],
                        choices=["best", "current_best"],
                        help="Choose metrics that should be recorded during "
                             "optimization. These will also be plotted if possible.")

    parser.add_argument("--min-sigma", type=float,
                        help="If set, the ES will stop after its sigma value drops"
                             "below this amount.")

    parser.add_argument("--no-progress", type=int,
                        help="If set, the algorithm will stop if it has no progress"
                             "for this many generations.")

    parser.add_argument("--max-seconds", type=int,
                        help="If set, the algorithm will stop after this many seconds.")

    parser.add_argument("-p", "--plot-progress", action="store_true",
                        help="This flag controls whether to plot progress using "
                             "matplotlib. This has only been tested using the "
                             "TkAgg backend.")

    parser.add_argument("-N", "--n_runs", type=int, default=1,
                        help="Run this algorithm N times. If N>1, print mean and "
                             "standard deviation of best results. (Default 1)")

    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Set verbosity level")

    args = parser.parse_args()

    if args.n_runs > 1 and args.plot_progress:
        warnings.warn("Running multiple experiments with plot_progress==True is not recommended")

    results = []

    for i in range(args.n_runs):
        algorithm = _algorithm_from_args(args)
        algorithm.evolve()
        results.append(algorithm.best.score)
        if args.n_runs > 1:
            print(f"Run {i}: {algorithm.best.score:.2f}")

    if args.n_runs > 1:
        print()
        print(f"All runs completed.")
        print(f"Average score:        {mean(results):.2f}")
        print(f"Standard deviation:   {stdev(results):.2f}")
    else:
        print(f"Best score: {results[0]}")

    print(f"Best known result: {algorithm.problem.best_known}")


if __name__ == "__main__":
    main()
