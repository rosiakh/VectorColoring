""" Main file used to run experiments. """

from configuration import algorithm_instances_config
from graph.graph_io import *
from run.algorithm_runner import run_check_save_on_graphs

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

    graphs = read_graphs_from_directory(graphs_directory)

    run_check_save_on_graphs(graphs, algorithms)
