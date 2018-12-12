"""Script for running the algorithms on specified graphs.

Usage: python test.py
"""

from algorithm import *
from graph_io import *
from graph_create import *


def display_colorings(colorings):
    """Displays results of the coloring algorithms to stdout."""
    for g in colorings:
        print '\nGraph: {0}'.format(g.name)
        for (alg_name, alg_coloring) in colorings[g]:
            if (not check_if_coloring_legal(g, alg_coloring)):
                print '\tColoring obtained by \n\t\t{0} \n\tis not legal'.format(alg_name)
            else:
                print '\tColoring obtained by \n\t\t{0} \n\tis legal and uses {1} colors\n'.format(
                    alg_name, str(len(set(alg_coloring.values()))))


# Logging configuration
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Test graph creation
graphs = []
graphs.append(read_graph_from_file("dimacs", "DSJC125.5", starting_index=1))
# graphs.append(read_graph_from_file("dimacs", "DSJC500.1", starting_index=1))

# graphs.append(create_k_cycle(40, 120))
# graphs.append(create_random_graph(150, 10000))

# Test algorithm creation
algorithms = []
# algorithms.append(VectorColoringAlgorithm(
#     partial_color_strategy='hyperplane_partition',
#     partition_strategy='orthogonal',
#     wigderson_strategy='no_wigderson',
#     alg_name='orthogonal hyperplane partition'
# ))
algorithms.append(VectorColoringAlgorithm(
    partial_color_strategy='vector_projection',
    find_ind_sets_strategy='clustering',
    wigderson_strategy='no_wigderson',
    alg_name='clustering no wigderson'
))

# algorithms.append(VectorColoringAlgorithm(
#     partial_color_strategy='hyperplane_partition',
#     partition_strategy='clustering',
#     wigderson_strategy='no_wigderson',
#     alg_name='clustering hyperplane partition no wigderson'
# ))
# algorithms.append(VectorColoringAlgorithm(
#     partial_color_strategy='hyperplane_partition',
#     partition_strategy='random',
#     wigderson_strategy='no_wigderson',
#     alg_name='random hyperplane partition no wigderson'
# ))
# algorithms.append(VectorColoringAlgorithm(
#     partial_color_strategy='vector_projection',
#     find_ind_sets_strategy='random_vector_projection',
#     wigderson_strategy='recursive_wigderson',
#     alg_name='random vector projection recursive wigderson'
# ))
# algorithms.append(VectorColoringAlgorithm(
#     partial_color_strategy='vector_projection',
#     find_ind_sets_strategy='random_vector_projection',
#     wigderson_strategy='no_wigderson',
# alg_name='random vector projection no wigderson'
# ))
algorithms.append(ColoringAlgorithm(
    lambda g: nx.algorithms.coloring.greedy_color(g, strategy='independent_set'), 'greedy_independent_set'))
algorithms.append(ColoringAlgorithm(
    lambda g: nx.algorithms.coloring.greedy_color(g, strategy='DSATUR'), 'dsatur'))

# Run algorithms to obtain colorings
colorings = {}
for g in graphs:
    colorings[g] = []
    for alg in algorithms:
        colorings[g].append((alg.get_algorithm_name(), alg.color_graph(g, verbose=False)))

logging.shutdown()

# Check if colorings are legal
for g in colorings:
    for (alg_name, alg_coloring) in colorings[g]:
        if not check_if_coloring_legal(g, alg_coloring):
            raise Exception('Coloring obtained by {0} on {1} is not legal'.format(alg_name, g.name))

display_colorings(colorings)
