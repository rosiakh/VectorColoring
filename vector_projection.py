import logging
import math

import networkx as nx

from algorithm_helper import *


def partially_color_graph_by_vector_projections(g, L, colors, find_ind_sets_strategy):
    """This strategy finds one or more independent set finding it one list of sets at a time."""

    def better_ind_sets(g, ind_sets1, ind_sets2):
        """Returns true if the first list of independent sets is better than the second."""

        if ind_sets2 is None or len(ind_sets2) == 0:
            return True

        if ind_sets1 is None or len(ind_sets1) == 0:
            return False

        temp_colors1 = {v: -1 for s in ind_sets1 for v in s}
        temp_colors2 = {v: -1 for s in ind_sets2 for v in s}

        clr = 0
        for ind_set in ind_sets1:
            clr += 1
            for v in ind_set:
                temp_colors1[v] = clr

        clr = 0
        for ind_set in ind_sets2:
            clr += 1
            for v in ind_set:
                temp_colors2[v] = clr

        avg1 = len(temp_colors1.keys()) / float(len(set(temp_colors1.values())))
        avg2 = len(temp_colors2.keys()) / float(len(set(temp_colors2.values())))

        # It is not optimal as there may be some set that is completely contained in sum of other sets and only
        #   by coloring it first, will we spare its color

        # set_sizes1 = map(lambda s: len(s), ind_sets1)
        # set_sizes2 = map(lambda s: len(s), ind_sets2)
        # avg1 = float(sum(set_sizes1)) / len(set_sizes1)
        # avg2 = float(sum(set_sizes2)) / len(set_sizes2)

        return avg1 > avg2

    def update_colors_and_graph(g, colors, ind_sets):

        color = max(colors.values())
        for ind_set in ind_sets:
            color += 1
            for v in ind_set:
                if colors[v] == -1:
                    colors[v] = color
            g.remove_nodes_from(ind_set)

        logging.info('There are {0} vertices left to color'.format(g.number_of_nodes()))

    logging.info('Looking for independent sets using vector projections strategy...')

    # Config
    iterations = 10

    best_ind_sets = None
    for it in range(iterations):
        ind_sets = find_ind_sets_strategy(g, L)  # Returns list of sets

        if better_ind_sets(g, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets

    update_colors_and_graph(g, colors, best_ind_sets)
    logging.debug('Found independent sets (maybe identical) of sizes: ' + str([len(s) for s in best_ind_sets]))


def find_ind_set_by_random_vector_projection(g, L):
    """Returns one set of vertices (one-element list) obtained by random vector projection."""

    def better_subgraph(subg_v1, subg_e1, subg_v2, subg_e2):
        """Returns True if subg1 is a better subgraph than subg2. Subgraph1 is represented by its vertices and edges.
            subg_v1 and subg_e1."""

        # TODO: Think about this code. Seems too short. Maybe can use 'fix_nodes' function

        if subg_v2 is None or len(subg_v2) == 0:
            return True

        if subg_v1 is None or len(subg_v1) == 0:
            return False

        subg1 = nx.Graph()
        subg1.add_nodes_from(subg_v1)
        subg1.add_edges_from(subg_e1)

        subg2 = nx.Graph()
        subg2.add_nodes_from(subg_v2)
        subg2.add_edges_from(subg_e2)

        to_del1 = nodes_to_delete(subg1, {v: 0 for v in subg1.nodes()})
        to_del2 = nodes_to_delete(subg2, {v: 0 for v in subg2.nodes()})

        subg1.remove_nodes_from(to_del1)
        subg2.remove_nodes_from(to_del2)

        return subg1.number_of_nodes() > subg2.number_of_nodes()

    def compute_c_opt(g, L):
        """Computes optimal c parameter according to KMS.

        Args:
            g (nx.Graph): Graph
            L (2-dim array): Vector coloring of g
        """

        max_degree = max(dict(g.degree()).values())
        k = find_number_of_vector_colors_from_vector_coloring(g, L)
        temp = (2 * (k - 2) * math.log(max_degree)) / k
        if temp >= 0:
            c = math.sqrt(temp)
        else:
            c = 0.0

        return c

    # Config
    iterations = 100
    c = 0.7 * compute_c_opt(g, L)

    n = g.number_of_nodes()
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(g.nodes()))}

    # TODO: Maybe stop searching only if for last 100 we haven't found any better set (if we have one already)

    best_subgraph_edges = None
    best_subgraph_nodes = None
    for it in range(iterations):
        r = np.random.normal(0, 1, n)
        x = np.dot(L, r)
        current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c}
        current_subgraph_edges = {(i, j) for i, j in g.edges() if
                                  (i in current_subgraph_nodes and j in current_subgraph_nodes)}

        if better_subgraph(current_subgraph_nodes, current_subgraph_edges, best_subgraph_nodes, best_subgraph_edges):
            best_subgraph_nodes = current_subgraph_nodes
            best_subgraph_edges = current_subgraph_edges

    subgraph = nx.Graph()
    subgraph.add_nodes_from(best_subgraph_nodes)
    subgraph.add_edges_from(best_subgraph_edges)
    temp_colors = {v: 0 for v in subgraph.nodes()}

    nodes_to_del = nodes_to_delete(subgraph, temp_colors, strategy='max_degree_first')

    subgraph.remove_nodes_from(nodes_to_del)

    return [set(subgraph.nodes())]

    # if subgraph.number_of_nodes() == 0:
    #     return [set()]
    #
    # # Remove one node from each edge to obtain an independent set
    # nodes_by_degree = get_nodes_sorted_by_degree(subgraph)
    #
    # while nodes_by_degree:
    #     if subgraph.degree(nodes_by_degree[0]) == 0:
    #         break
    #     subgraph.remove_node(nodes_by_degree[0])
    #     nodes_by_degree = get_nodes_sorted_by_degree(subgraph)
    #
    # # All nodes have degree 0 so they are independent set.
    # return [set(nodes_by_degree)]


def find_multiple_ind_sets_by_random_vector_projections(g, L):
    iterations = 100
    nr_of_sets = 4

    it = 0
    ind_sets = []
    while len(ind_sets) < nr_of_sets and it < iterations:
        ind_sets.extend(find_ind_set_by_random_vector_projection(g, L))

    return ind_sets

# def partially_color_graph_by_vector_projection_strategy(g, L, colors):
#     """Colors some of the vertices of the graph given it's vector coloring.
#
#         Vertices are colored using partition of the space by vector projections. When parameters are set correctly then
#             with high probability at least half of the vertices should be colored i.e. their values in 'colors' dictionary
#             should be set to some non-negative number.
#
#         In
#
#         Args:
#             g (Graph): Graph to be colored. It is modified so that only not colored vertices are left.
#             L (2-dim matrix): Rows of L are vector coloring of g -
#                 nth row is assigned to nth vertex which doesn't have to be vertex 'n'.
#             colors (dict): Global vertex-color dictionary.
#             M (2-dim matrix): Matrix coloring of g.
#             iterations (int): Number of random vectors (also potentially ind. sets) considered
#         """
#
#     def compute_c_opt_parameter(G, L):
#         """Computes optimal c parameter according to KMS.
#
#         Args:
#             g (nx.Graph): Graph
#             L (2-dim array): Vector coloring of g
#         """
#
#         max_degree = max(dict(G.degree()).values())
#         k = find_number_of_vector_colors_from_vector_coloring(G, L)
#         temp = (2 * (k - 2) * math.log(max_degree)) / k
#         if temp >= 0:
#             c = math.sqrt(temp)
#         else:
#             c = 0.0
#
#         return c
#
#     def get_nodes_sorted_by_degree(g):
#         """Returns list of nodes sorted by degree in descending order."""
#
#         return [item[0] for item in
#                 sorted(list(g.degree(g.nodes())), key=lambda item: item[1], reverse=True)]
#
#     def vector_projection_options(g, L, colors):
#
#         max_degree = max(dict(g.degree()).values())
#         k = find_number_of_vector_colors_from_vector_coloring(g, L)
#         temp = (2 * (k - 2) * math.log(max_degree)) / k
#         if temp >= 0:
#             c_opt = math.sqrt(temp)
#         else:
#             c_opt = 0.0
#
#         iterations = int(min(max(n ** 2, 1e3), 1e3))
#
#     logging.info('Looking for independent set using vector projection strategy...')
#
#     nr_of_c_values = 10
#     n = g.number_of_nodes()
#     inv_vertices_mapping = {i: v for i, v in enumerate(sorted(g.nodes()))}
#     c_opt = compute_c_opt_parameter(g, L)
#     iterations = int(min(max(n ** 2, 1e3), 1e3))
#
#     r = np.random.normal(0, 1, n)
#     x = np.dot(L, r)
#     best_subgraph_nodes = [inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c_opt]
#     best_subgraph_edges = {(i, j) for i, j in g.edges() if
#                            (i in best_subgraph_nodes and j in best_subgraph_nodes)}
#
#     logging.debug('Vector projections to consider: {0} x {1}'.format(nr_of_c_values, iterations))
#
#     for c in np.linspace(c_opt / 2, 3 * c_opt / 2, nr_of_c_values):
#         it = 0
#         while it < iterations:
#             it += 1
#             r = np.random.normal(0, 1, n)
#             x = np.dot(L, r)
#             initial_subgraph_nodes = [inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c]
#             initial_subgraph_edges = {(i, j) for i, j in g.edges() if
#                                       (i in initial_subgraph_nodes and j in initial_subgraph_nodes)}
#
#             current_difference = len(initial_subgraph_nodes) - len(initial_subgraph_edges)
#             if current_difference > len(best_subgraph_nodes) - len(best_subgraph_edges):
#                 best_subgraph_nodes = initial_subgraph_nodes
#                 best_subgraph_edges = initial_subgraph_edges
#
#     subgraph = nx.Graph()
#     subgraph.add_nodes_from(best_subgraph_nodes)
#     subgraph.add_edges_from(best_subgraph_edges)
#
#     if subgraph.number_of_nodes() == 0:
#         logging.info('Didn\'t manage to find any independent set')
#         return
#
#     # remove one node from each edge to obtain an independent set
#     nodes_by_degree = get_nodes_sorted_by_degree(subgraph)
#
#     while nodes_by_degree:
#         if subgraph.degree(nodes_by_degree[0]) == 0:
#             break
#         subgraph.remove_node(nodes_by_degree[0])
#         nodes_by_degree = get_nodes_sorted_by_degree(subgraph)
#
#     # All nodes have degree 0 so they are independent set. Color it using next free color and remove from g.
#     if nodes_by_degree:
#         min_available_color = max(colors.values()) + 1
#         for v in subgraph.nodes():
#             colors[v] = min_available_color
#         g.remove_nodes_from(subgraph.nodes())
#
#     logging.info('Found independent set of size {0}'.format(subgraph.number_of_nodes()))
