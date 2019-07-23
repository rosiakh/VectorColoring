""" Configuration needed to perform tests. """

from configuration import algorithm_instances_config

test_graphs_directory = '../resources/test_instances/'

test_algorithms = [
    algorithm_instances_config.algorithms_configured['IndSets strategy: random vector projection'],
    algorithm_instances_config.algorithms_configured['IndSets strategy: clustering'],
    algorithm_instances_config.algorithms_configured['Color and fix: random hyperplanes'],
    algorithm_instances_config.algorithms_configured['Color and fix: orthonormal hyperplanes'],
    algorithm_instances_config.algorithms_configured['Color and fix: clustering'],
    algorithm_instances_config.algorithms_configured['Greedy: DSATUR'],
    algorithm_instances_config.algorithms_configured['Greedy: Independent Set']
]
