import datetime
import json
import os
from os import listdir
from os.path import isfile, join

import networkx as nx

from algorithm import VectorColoringAlgorithm

dirname = '/home/hubert/VectorColoring/'
if not os.path.exists(dirname):
    os.makedirs(dirname)

vc_dirname = dirname + 'VectorColorings/'
if not os.path.exists(vc_dirname):
    os.makedirs(vc_dirname)


class RunResults:
    pass


def save_algorithm_run_data_to_file(algorithms_results):
    """Saves all relevant data_to_save from each algorithm run to file.

    Args:
        algorithms_results (dict): Key: nx.Graph - value: list of RunResults.
    """

    folder_name = vc_dirname + 'algorithm_run_' + datetime.datetime.now().strftime("%m-%d_%H-%M-%S") + '/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for graph in algorithms_results:
        filename = graph.name + '_' + datetime.datetime.now().strftime("%M-%S") + '.json'
        with open(folder_name + filename, 'w') as outfile:
            data_to_save = []
            for results in algorithms_results[graph]:  # One result per algorithm
                algorithm_data_to_save = {
                    'graph_name': graph.name,
                    'graph_family': graph.name,  # TODO
                    'graph_nr_of_vertices': graph.number_of_nodes(),
                    'graph_density': nx.classes.density(graph),
                    'avg_nr_of_colors': results.average_nr_of_colors,
                    'min_nr_of_colors': len(set(results.best_coloring.values())),
                    'algorithm_name': results.algorithm.get_algorithm_name(),
                    'avg_time': results.average_time,
                    'params': 'params' if isinstance(results.algorithm, VectorColoringAlgorithm) else 'na',  # TODO
                    'init_params': results.algorithm._literal_init_params if
                    isinstance(results.algorithm, VectorColoringAlgorithm) else 'na'
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


display_results_on_console('02-07_18-50-58')
