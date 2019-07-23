import json
import os
from os import listdir
from os.path import isfile, join

from configuration import paths_config
from data_to_save import DataToSave


class RunResults:
    """ Represents all the information that is gathered during run of a single algorithm on a single graph (possibly
        repeatedly). """

    def __init__(self, graph, algorithm, average_time, best_coloring, average_nr_of_colors, repetitions):
        self.graph = graph
        self.algorithm = algorithm
        self.average_time = average_time
        self.best_coloring = best_coloring
        self.average_nr_of_colors = average_nr_of_colors
        self.repetitions = repetitions


def get_sorted_graph_names(results):
    """ results is a dictionary (key is graph name) of lists of DataToSave objects. """

    return [graph_name for graph_name in sorted(results.keys(), key=lambda graph_name: (
        results[graph_name][0]['graph_family'],
        int(results[graph_name][0]['graph_nr_of_vertices']),
        float(results[graph_name][0]['graph_density'])))]


def get_sorted_algorithm_names(results):
    return [result['algorithm_name'] for result in results[results.keys()[0]]]


def save_runs_data_to_file(algorithms_results, subdir):
    """ algorithm_results is a graph -> list or RunResult dictionary. """

    for graph in algorithms_results:
        save_graph_run_data_to_file(algorithms_results[graph], graph, subdir)


def save_graph_run_data_to_file(graph_results, graph, subdir):
    """Saves results of all algorithms on one graph.

    Args:
        graph_results (list of RunResults): algorithm results for graph
    """

    filename = graph.name + '.json'
    directory = paths_config.results_directory() + "/" + subdir + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + filename, 'w') as outfile:
        data_to_save_for_graph = []

        for algorithm_run_result in graph_results:
            algorithm_data_to_save = DataToSave(graph, algorithm_run_result)
            data_to_save_for_graph.append(algorithm_data_to_save)

        json.dump(
            data_to_save_for_graph, outfile, ensure_ascii=False, indent=4, sort_keys=True, default=lambda d: d.__dict__)


def load_algorithm_run_data_from_seed(run_seed=None):
    directory = paths_config.results_directory(run_seed)
    return load_algorithm_run_data_from_directory(directory)


def load_algorithm_run_data_from_directory(directory):
    only_files = [f for f in listdir(directory) if isfile(join(directory, f))]

    graph_results = {}
    for filename in only_files:
        with open(directory + "/" + filename) as infile:
            graph_algorithms_results = json.load(infile)  # each file a list of data for each algorithm (not RunResults)
            graph_results[os.path.splitext(filename)[0]] = graph_algorithms_results

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
