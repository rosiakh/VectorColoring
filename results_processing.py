import json
import os
from os import listdir
from os.path import isfile, join

import networkx as nx

import config
from algorithm import VectorColoringAlgorithm


class RunResults:

    def __init__(self):
        self.graph = None
        self.algorithm = None
        self.average_time = -1
        self.best_coloring = None
        self.average_nr_of_colors = -1
        self.repetitions = -1


def save_graph_run_data_to_file(graph_results, graph):
    """Saves results of one algorithm on one graph.

    Args:
        graph_results (list of RunResults): algorithm results for graph
    """

    filename = graph.name + '.json'
    with open(config.vector_colorings_directory() + filename, 'w') as outfile:
        data_to_save = []
        for algorithm_run_data in graph_results:
            algorithm_data_to_save = {
                'graph_name': graph.name,
                'graph_family': graph.name,  # TODO
                'graph_nr_of_vertices': graph.number_of_nodes(),
                'graph_density': nx.classes.density(graph),
                'avg_nr_of_colors': algorithm_run_data.average_nr_of_colors,
                'min_nr_of_colors': len(set(algorithm_run_data.best_coloring.values())),
                'algorithm_name': algorithm_run_data.algorithm.get_algorithm_name(),
                'avg_time': algorithm_run_data.average_time,
                'params': 'params' if isinstance(algorithm_run_data.algorithm, VectorColoringAlgorithm) else 'N/A',
                # TODO
                'init_params': algorithm_run_data.algorithm._literal_init_params if
                isinstance(algorithm_run_data.algorithm, VectorColoringAlgorithm) else 'N/A'
            }
            data_to_save.append(algorithm_data_to_save)
        json.dump(data_to_save, outfile, ensure_ascii=False, indent=4, sort_keys=True)


def load_algorithm_run_data_from_file(run_datetime_str):
    folder_name = vc_dirname + 'algorithm_run_' + run_datetime_str + '/'
    onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]

    graph_results = {}
    for filename in onlyfiles:
        with open(folder_name + filename) as infile:
            graph_algorithms_results = json.load(infile)  # each file a list of data for each algorithm (not RunResults)
            graph_results[os.path.splitext(filename)[0]] = graph_algorithms_results
            a = 2

    # graph_results should be used as a basis for display
    return graph_results


def display_results_on_console(run_datetime_str):
    graph_results = load_algorithm_run_data_from_file(run_datetime_str)

    for graph_name in graph_results:  # graph_results - a list of dictionaries (one dict per algorithm)
        print 'Graph: {0}'.format(graph_name)
        for results in graph_results[graph_name]:
            print '\talgorithm: {0:50} min colors: {1}'.format(
                results['algorithm_name'], results['min_nr_of_colors'])
