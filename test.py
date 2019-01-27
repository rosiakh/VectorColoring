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

# graphs.append(nx.powerlaw_cluster_graph(180, 37, 0.3)) # duza przewaga dsatur

# graphs.append(create_erdos_renyi_graph(100, 0.5)) # nie widac roznicy
# graphs.append(nx.ring_of_cliques(10, 10))
# graphs.append(nx.connected_caveman_graph(10,10))
# graphs.append(nx.random_regular_graph(19, 160))
# graphs.append(nx.circular_ladder_graph(140))
# graphs.append(nx.dorogovtsev_goltsev_mendes_graph(2))
# graphs.append(nx.full_rary_tree(40, 350))
# graphs.append(nx.ladder_graph(150))
# graphs.append(nx.lollipop_graph(20, 50))
# graphs.append(nx.star_graph(150))
# graphs.append(nx.turan_graph(150, 9))
# graphs.append(nx.wheel_graph(170))
# graphs.append(nx.margulis_gabber_galil_graph(12))
# graphs.append(nx.chordal_cycle_graph(131))
# graphs.append(create_crown_graph(130))
# graphs.append(nx.triangular_lattice_graph(30,21))
# graphs.append(nx.tutte_graph())
# graphs.append(nx.random_lobster(130, 0.8, 0.7))
# graphs.append(nx.duplication_divergence_graph(170, 0.999))
# graphs.append(nx.geographical_threshold_graph(150, 0.2))
# graphs.append(nx.windmill_graph(16, 14))
# graphs.append(nx.mycielski_graph(8))
# graphs.append(nx.random_partition_graph([x for x in range(15, 23)], 0.9, 0.2))

# graphs.append(read_graph_from_file('other', 'grotzsch', starting_index=0))
# graphs.append(read_graph_from_file("dimacs", "DSJC125.1", starting_index=1))
graphs.append(read_graph_from_file("dimacs", "DSJC1000.1", starting_index=1))

# graphs.append(create_k_cycle(4, 20))
# graphs.append(create_erdos_renyi_graph_edges(20, 5*13))

# Test algorithm creation
algorithms = []
# algorithms.append(VectorColoringAlgorithm(
#     partial_color_strategy='hyperplane_partition',
#     partition_strategy='orthogonal',
#     wigderson_strategy='no_wigderson',
#     alg_name='orthogonal hyperplane partition'
# ))
# algorithms.append(VectorColoringAlgorithm(
#     partial_color_strategy='vector_projection',
#     find_ind_sets_strategy='clustering',
#     wigderson_strategy='no_wigderson',
#     alg_name='clustering no wigderson'
# ))
# algorithms.append(VectorColoringAlgorithm(
#     partial_color_strategy='hyperplane_partition',
#     partition_strategy='kmeans_clustering',
#     wigderson_strategy='no_wigderson',
#     alg_name='kmeans clustering partition'
# ))
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
algorithms.append(VectorColoringAlgorithm(
    partial_color_strategy='vector_projection',
    find_ind_sets_strategy='random_vector_projection_kms_prim',
    wigderson_strategy='no_wigderson',
    alg_name='random vector projection no wigderson'
))
algorithms.append(ColoringAlgorithm(
    lambda g: nx.algorithms.coloring.greedy_color(g, strategy='independent_set'), 'greedy_independent_set'))
algorithms.append(ColoringAlgorithm(
    lambda g: nx.algorithms.coloring.greedy_color(g, strategy='DSATUR'), 'dsatur'))
# algorithms.append(ColoringAlgorithm(lambda g: compute_optimal_coloring_lp(g), 'optimal_coloring_lp'))
#algorithms.append(ColoringAlgorithm(lambda g: compute_optimal_coloring_dp(g), 'optimal_coloring_dp'))

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
