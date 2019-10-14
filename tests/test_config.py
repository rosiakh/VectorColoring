""" Configuration needed to perform tests. """

from configuration import algorithm_instances_config

test_graphs_directory = '../resources/test_instances/'

# test_algorithms = [
# algorithm_instances_config.algorithms_configured['IndSets strategy -random vector projection -linspace'],
# algorithm_instances_config.algorithms_configured['IndSets strategy -random vector projection -ratio'],
# algorithm_instances_config.algorithms_configured['IndSets strategy -dummy vector coloring-projection -linspace'],
# algorithm_instances_config.algorithms_configured['IndSets strategy -dummy vector coloring-projection -ratio'],
# algorithm_instances_config.algorithms_configured['IndSets strategy -dummy vector coloring -greedy -all-vertices'],
# algorithm_instances_config.algorithms_configured['IndSets strategy -dummy vector coloring -greedy -first-edge'],
# algorithm_instances_config.algorithms_configured['IndSets strategy -dummy vector coloring -greedy -ratio'],
# algorithm_instances_config.algorithms_configured['IndSets strategy -clustering'],
# algorithm_instances_config.algorithms_configured['Color and fix -random hyperplanes'],
# algorithm_instances_config.algorithms_configured['Color and fix -orthonormal hyperplanes'],
# algorithm_instances_config.algorithms_configured['Color and fix -clustering'],
# algorithm_instances_config.algorithms_configured['Greedy - DSATUR'],
# algorithm_instances_config.algorithms_configured['Greedy - Independent Set']
# ]

run_algorithms = [
    algorithm_instances_config.algorithms_configured['IndSets strategy -clustering'],
    algorithm_instances_config.algorithms_configured['Color and fix -clustering'],
    algorithm_instances_config.algorithms_configured['Color and fix -random hyperplanes'],
    algorithm_instances_config.algorithms_configured[
        'IndSets strategy -random vector projection -best-many-trials_25x25'],  # TEN BYL WYBRANY JAKO DRUGI
    algorithm_instances_config.algorithms_configured['IndSets strategy -random vector projection -linspace'],
    algorithm_instances_config.algorithms_configured['IndSets strategy -random vector projection -ratio -low'],
    algorithm_instances_config.algorithms_configured['IndSets strategy -random vector projection -ratio -high'],
    algorithm_instances_config.algorithms_configured[
        'IndSets strategy -dummy vector coloring -projection -linspace -1_c_param_per_vector'],
    algorithm_instances_config.algorithms_configured[
        'IndSets strategy -dummy vector coloring -projection -linspace -10_c_param_per_vector'],  #TEN BYL WYBRANY
    algorithm_instances_config.algorithms_configured[
        'IndSets strategy -dummy vector coloring -projection -ratio -fast'],
    algorithm_instances_config.algorithms_configured[
        'IndSets strategy -dummy vector coloring -projection -ratio -slow'],
    algorithm_instances_config.algorithms_configured['IndSets strategy -dummy vector coloring -greedy -all-vertices'],
    algorithm_instances_config.algorithms_configured['IndSets strategy -dummy vector coloring -greedy -first-edge'],
    algorithm_instances_config.algorithms_configured['IndSets strategy -dummy vector coloring -greedy -ratio -high'],
    algorithm_instances_config.algorithms_configured['IndSets strategy -dummy vector coloring -greedy -ratio -low'],
    algorithm_instances_config.algorithms_configured['Greedy -Independent Set'],
    algorithm_instances_config.algorithms_configured['Greedy -DSATUR'],
]
