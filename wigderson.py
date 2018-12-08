import networkx.algorithms.approximation
import networkx as nx
import logging
import algorithm
from algorithm_helper import *


def max_ind_set_wigderson_strategy(g, colors, L):
    """Colors some of the nodes using Wigderson technique.

        Args:
            g (graph): Graph to be processed.
            colors (dict): Global vertex-color dictionary.
            threshold_degree (int): Lower boundary for degree of processed nodes of g.
        """

    print '\n starting Wigderson algorithm:'
    print '     threshold deg:', threshold_degree
    max_degree = max(dict(g.degree()).values())
    iterations = 0
    while max_degree > threshold_degree:
        iterations += 1
        print '\n iteration', iterations
        print ' max deg:', max_degree

        # Find any vertex with degree equal to max_degree.
        max_vertex = 0
        for v in dict(g.degree()):
            if g.degree()[v] == max_degree:
                max_vertex = v
                break

        neighbors_subgraph = g.subgraph(g.neighbors(max_vertex))

        # Find large independent set in neighbors subgraph using approximate maximum independent set algorithm.
        # Can we find this large independent set using our algorithm recursively?
        min_available_color = max(colors.values()) + 1
        max_ind_set = nx.algorithms.approximation.maximum_independent_set(neighbors_subgraph)
        # max_ind_set = nx.maximal_independent_set(neighbors_subgraph)
        for v in max_ind_set:
            colors[v] = min_available_color

        # Remove nodes that have just been colored
        g.remove_nodes_from(max_ind_set)
        max_degree = max(dict(g.degree()).values())

    return


def recursive_wigderson_strategy(g, colors, L):
    """Colors some of the nodes using Wigderson technique.

    Args:
        g (graph): Graph to be processed.
        colors (dict): Global vertex-color dictionary.
        threshold_degree (int): Lower boundary for degree of processed nodes of g.
    """

    # TODO; Maybe try iterative version

    logging.info('Starting Wigderson technique...')

    k = find_number_of_vector_colors_from_vector_coloring(g, L)
    n = g.number_of_nodes()
    max_degree = max(dict(g.degree()).values())
    threshold_degree = pow(n, (k - 1) / k)
    it = 0
    while max_degree > threshold_degree:
        it += 1

        # Find any vertex with degree equal to max_degree.
        max_vertex = 0
        for v in dict(g.degree()):
            if g.degree()[v] == max_degree:
                max_vertex = v
                break

        neighbors_subgraph = nx.Graph()
        neighbors_subgraph.add_nodes_from(g.neighbors(max_vertex))
        neighbors_subgraph.add_edges_from(g.subgraph(g.neighbors(max_vertex)).edges())
        alg = algorithm.VectorColoringAlgorithm(
            partial_color_strategy='vector_projection',
            find_ind_sets_strategy='random_vector_projection',
            wigderson_strategy='recursive_wigderson'
        )
        neighbors_colors = alg.color_graph(neighbors_subgraph)
        for v in neighbors_subgraph.nodes():
            colors[v] = neighbors_colors[v]

        # Remove nodes that have just been colored
        g.remove_nodes_from(neighbors_subgraph.nodes())
        max_degree = max(dict(g.degree()).values())

    return


def no_wigderson_strategy(g, colors, L):
    pass
