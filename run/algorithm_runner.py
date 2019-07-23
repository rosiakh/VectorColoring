"""Script for running the algorithms on specified graphs.

Usage: python algorithm_runner.py
"""

from timeit import default_timer as timer

from coloring.algorithm import *
from configuration import run_config
from graph.graph_io import *
from result_processing.result_processing import *


def run_check_save_on_directory(algorithms, directory):
    graphs = read_graphs_from_directory(directory)
    split_subdir = directory.split("/")
    results_subdir = split_subdir[len(split_subdir) - 1]
    return run_check_save_on_graphs(graphs, algorithms, results_subdir)


def run_check_save_on_graphs(graphs, algorithms, results_subdir):
    algorithms_results = run(graphs, algorithms)

    for graph in algorithms_results:
        for run_results in algorithms_results[graph]:
            if not check_if_coloring_legal(graph, run_results.best_coloring):
                raise Exception(
                    'Coloring obtained by {0} on {1} is not legal'.format(run_results.algorithm.name, graph.name))

    save_runs_data_to_file(algorithms_results, results_subdir)


def run(graphs, algorithms):
    """Runs each algorithm on each graph."""

    logging.basicConfig(format='%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

    algorithms_results = {}  # Dictionary - graph: list of RunResults (one result per algorithm)

    for graph_counter, graph in enumerate(graphs):
        algorithms_results[graph] = []

        for alg_counter, alg in enumerate(algorithms):
            logging.info("\nComputing graph: {0} ({2}/{3}), algorithm: {1} ({4}/{5}) ...\n".format(
                graph.name, alg.get_algorithm_name(), graph_counter + 1, len(graphs), alg_counter + 1, len(algorithms)))

            times = []
            graph_colorings = []

            for _ in range(run_config.repetitions_per_graph):
                start = timer()
                coloring = alg.color_graph(graph)
                end = timer()
                times.append(end - start)
                graph_colorings.append(coloring)

            run_results = RunResults(
                graph=graph,
                algorithm=alg,
                average_time=np.mean(times),
                best_coloring=min(graph_colorings, key=lambda coloring: len(set(coloring.values()))),
                average_nr_of_colors=np.mean([len(set(coloring.values())) for coloring in graph_colorings]),
                repetitions=run_config.repetitions_per_graph
            )

            algorithms_results[graph].append(run_results)
            logging.info("Done graph: {0}, algorithm: {1}, colors: {2}, time: {3:6.2f} s ...\n".format(
                graph.name, alg.get_algorithm_name(), len(set(run_results.best_coloring.values())),
                run_results.average_time))

    return algorithms_results
