import json
import os
from os import listdir
from os.path import isfile, join, isdir

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from configuration import paths_config
from data_to_save import DataToSave


class RunResults:
    """Represents all the information that is gathered during run of a single algorithm on a single graph (possibly
        repeatedly). """

    def __init__(self, graph, algorithm, average_time, best_coloring, average_nr_of_colors, repetitions):
        self.graph = graph
        self.algorithm = algorithm
        self.average_time = average_time
        self.best_coloring = best_coloring
        self.average_nr_of_colors = average_nr_of_colors
        self.repetitions = repetitions


def get_sorted_graph_names(results_directory):
    """Get sorted names of the graphs whose run results reside in results_directory by looking at the names of
        subdirectories contained in results_directory.

    :param results_directory: directory in which to look for run results
    :return: list of names of graphs whose results reside in results_directory
    """

    graph_directories = [d for d in listdir(results_directory) if isdir(join(results_directory, d))]
    return graph_directories


def get_sorted_algorithm_names(results):
    """

    :param results:
    :return: sorted
    """
    return results.keys()


def save_run_result_to_file(run_result, subdir):
    """ Saves result of one algorithm on one graph."""

    run_result_save_path, run_result_save_dir = \
        get_run_result_save_path(run_result.graph, run_result.algorithm, subdir)
    if not os.path.exists(run_result_save_dir):
        os.makedirs(run_result_save_dir)

    with open(run_result_save_path, 'w') as outfile:
        data_to_save_for_graph = []
        algorithm_data_to_save = DataToSave(run_result.graph, run_result)
        data_to_save_for_graph.append(algorithm_data_to_save)

        json.dump(
            data_to_save_for_graph, outfile, ensure_ascii=False, indent=4, sort_keys=True,
            default=lambda d: d.__dict__)


def get_run_result_save_path(graph, algorithm, results_subdir):
    """Find path that should be used to save run results for given graph, algorithm and results_subdir

    :param graph: (nx.Graph)
    :param algorithm: coloring algorithm
    :param results_subdir:
    :return: created path to save run results
    """

    graph_dirname = graph.name.replace(":", "")
    directory = paths_config.results_directory() + "/" + results_subdir + "/" + graph_dirname + "/"
    filename = algorithm.get_algorithm_name() + ".json"
    filename = filename.replace(":", "")
    return os.path.join(directory, filename), directory


def save_runs_data_to_file(algorithms_results, subdir):
    """Saves algorithms_results to a file

    :param algorithms_results: a graph -> list or RunResult dictionary.
    :param subdir: subdirectory in which to save run results
    """

    for graph in algorithms_results:
        save_graph_run_data_to_file(algorithms_results[graph], graph, subdir)


def save_graph_run_data_to_file(graph_results, graph, subdir):
    """Saves results of all algorithms on one graph in:
        paths_config.results_directory()/subdir/graph_dirname/

    :param graph_results: (list of RunResults) algorithm results for graph
    :param graph: (nx.Graph)
    :param subdir: subdirectory in which to save run data
    """

    graph_dirname = graph.name.replace(":", "")
    directory = paths_config.results_directory() + "/" + subdir + "/" + graph_dirname + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for algorithm_run_result in graph_results:
        filename = algorithm_run_result.algorithm.get_algorithm_name() + ".json"
        filename = filename.replace(":", "")
        with open(directory + filename, 'w') as outfile:
            data_to_save_for_graph = []
            algorithm_data_to_save = DataToSave(graph, algorithm_run_result)
            data_to_save_for_graph.append(algorithm_data_to_save)

            json.dump(
                data_to_save_for_graph, outfile, ensure_ascii=False, indent=4, sort_keys=True,
                default=lambda d: d.__dict__)


def load_algorithm_run_data_from_seed(run_seed=None):
    """Load algorithm run data associated with given run seed

    :param run_seed: a run seed
    :return: algorithm run data associated with given run seed
    """
    directory = paths_config.results_directory(run_seed)
    return load_algorithm_run_data_from_results_directory(directory)


def load_algorithm_run_data_from_results_directory(directory):
    """Load algorithm run data from given directory

    :param directory: directory in which to look for algorithm run data
    :return: (map directory -> graph_results) loaded run data
    """

    graph_directories = [d for d in listdir(directory) if isdir(join(directory, d))]
    general_results = {}
    for graph_directory in graph_directories:
        full_graph_directory = join(directory, graph_directory)
        graph_results = load_algorithm_run_data_from_graph_directory(full_graph_directory)
        general_results[graph_directory] = graph_results

    return general_results


def load_algorithm_run_data_from_graph_directory(graph_directory):
    """Load algorithm run data from given graph directory

    :param graph_directory: directory in which to look for run data
    :return: (map)
    """

    only_files = [f for f in listdir(graph_directory) if isfile(join(graph_directory, f))]
    graph_results = {}
    for filename in only_files:
        with open(join(graph_directory, filename)) as infile:
            graph_algorithm_results = json.load(infile)  # a list of graph x algorithm results (usually has length 1)
            graph_results[os.path.splitext(filename)[0]] = graph_algorithm_results

    return graph_results


def read_chromatic_numbers_file():
    chromatic_numbers = {}
    exact = True
    chi_ind = -1
    chi_lb_ind = -1
    chi_ub_ind = -1

    if not os.path.exists(paths_config.chromatic_numbers_path):
        return None

    with open(paths_config.chromatic_numbers_path, 'r') as infile:
        for line in infile:
            values = line.split()
            if values[0] == 'x':
                if len(values) == 2:
                    exact = True
                    chi_ind = int(values[1])
                else:
                    exact = False
                    chi_lb_ind = int(values[1])
                    chi_ub_ind = int(values[2])
            else:
                if exact:
                    chromatic_numbers[values[0]] = str(values[chi_ind])
                else:
                    chromatic_numbers[values[0]] = '[' + values[chi_lb_ind] + ';' + values[chi_ub_ind] + ']'

    return chromatic_numbers


def draw_distributions(dummy_matrix_coloring, beta=1.0):
    draw_dummy_dot_products(dummy_matrix_coloring, beta)

    plt.legend()
    plt.show()


def draw_dummy_dot_products(dummy_matrix_coloring, beta_factor):
    n = dummy_matrix_coloring.shape[0]
    dummy_dot_products = dummy_matrix_coloring[n - 1][0:n - 1]

    sns.set(color_codes=True)
    sns.distplot(
        dummy_dot_products,
        label='beta: {0}\nmin: {1:.3f}\nmean: {2:.3f}\nvar: {3:.3f}'.format(
            beta_factor, np.min(dummy_dot_products), np.mean(dummy_dot_products), np.var(dummy_dot_products)),
        rug=True,
        norm_hist=True)
