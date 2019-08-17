""" Main file used to run experiments. """

import networkx as nx

from coloring.optimal_coloring import compute_optimal_coloring_dp
from configuration import algorithm_instances_config
from graph.graph_io import *

graphs_directory = '../resources/graph_instances/DIMACS/'

algorithms = [
    algorithm_instances_config.algorithms_configured['IndSets strategy: random vector projection'],
    algorithm_instances_config.algorithms_configured['IndSets strategy: clustering'],
    algorithm_instances_config.algorithms_configured['Color and fix: random hyperplanes'],
    algorithm_instances_config.algorithms_configured['Color and fix: orthonormal hyperplanes'],
    algorithm_instances_config.algorithms_configured['Color and fix: clustering']
]

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

    # graphs = read_graphs_from_directory(graphs_directory)
    #
    # run_check_save_on_graphs(graphs, algorithms)
    import os

    path = os.path.join(os.getcwd(), "resources/graph_instances/other/grotzsch.col")
    graph = read_graph_from_file(path)
    graph = nx.complete_graph(10)
    coloring = compute_optimal_coloring_dp(graph)
