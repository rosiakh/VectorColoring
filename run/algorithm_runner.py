"""Script for running the algorithms on specified graphs.

Usage: python algorithm_runner.py
"""

from timeit import default_timer as timer

from coloring.algorithm import *
from configuration import run_config
from graph.graph_io import *
from result_processing import result_io
from result_processing.result_processing import *


def run_check_save_on_all_subdirectories(algorithms, directory):
    create_algorithms_info_file(algorithms, directory)
    for subdir in result_io.all_subdirs_of(directory):
        run_check_save_on_directory(algorithms, subdir)


def create_algorithms_info_file(algorithms, directory):
    path = os.path.join(paths_config.results_directory(), paths_config.algorithm_info_filename)
    with open(path, 'w') as f:
        for algorithm in algorithms:
            f.write(algorithm.get_algorithm_name() + "\n")


def run_check_save_on_directory(algorithms, directory):
    graphs = read_graphs_from_directory(directory)
    split_subdir = directory.split(os.sep)
    results_subdir = split_subdir[len(split_subdir) - 1]
    return run_check_save_on_graphs(graphs, algorithms, results_subdir)


def run_check_save_on_graphs(graphs, algorithms, results_subdir):
    logging.basicConfig(format='%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

    sorted_and_pruned_graphs = sort_and_prune_graphs(graphs)
    for graph_counter, graph in enumerate(sorted_and_pruned_graphs):
        for algorithm_counter, algorithm in enumerate(algorithms):
            logging.info("\nSTARTING ON GRAPH: {0} ({2}/{3}), ALGORITHM: {1} ({4}/{5}) ...".format(graph.name,
                                                                                                   algorithm.get_algorithm_name(),
                                                                                                   graph_counter + 1,
                                                                                                   len(
                                                                                                       sorted_and_pruned_graphs),
                                                                                                   algorithm_counter + 1,
                                                                                                   len(algorithms)))

            # todo check if result already exists
            if not result_already_exists(graph, algorithm, results_subdir) or run_config.overwrite_results:
                try:
                    run_result = run_algorithm_on_graph(graph, algorithm)
                except Exception, e:
                    print 'Error has occurred during coloring: ' + str(e)
                    continue

                if not check_if_coloring_legal(graph, run_result.best_coloring):
                    raise Exception(
                        'Coloring obtained by {0} on {1} is not legal'.format(run_result.algorithm.name, graph.name))

                save_run_result_to_file(run_result, results_subdir)


def sort_and_prune_graphs(graphs):
    graphs.sort(key=lambda g: (g.number_of_nodes() + g.number_of_edges()) ** 2)
    pruned_graphs = [g for g in graphs if (g.number_of_nodes() + g.number_of_edges()) ** 2 < (350 + 15000) ** 2]
    return pruned_graphs


def result_already_exists(graph, algorithm, results_subdir):
    path, _ = get_run_result_save_path(graph, algorithm, results_subdir)
    return os.path.exists(path)


def run_algorithm_on_graph(graph, algorithm):
    times = []
    graph_colorings = []

    for _ in range(run_config.repetitions_per_graph):
        start = timer()
        coloring = algorithm.color_graph(graph)
        end = timer()
        times.append(end - start)
        graph_colorings.append(coloring)

    run_results = RunResults(
        graph=graph,
        algorithm=algorithm,
        average_time=np.mean(times),
        best_coloring=min(graph_colorings, key=lambda coloring: len(set(coloring.values()))),
        average_nr_of_colors=np.mean([len(set(coloring.values())) for coloring in graph_colorings]),
        repetitions=run_config.repetitions_per_graph
    )

    logging.info("Done graph: {0}, algorithm: {1}, colors: {2}, time: {3:6.2f} s ...\n".format(
        graph.name, algorithm.get_algorithm_name(), len(set(run_results.best_coloring.values())),
        run_results.average_time))

    return run_results


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
