import logging
import math

import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster

from algorithm_helper import *

params = {
    'nr_of_times_restarting_ind_set_strategy': 10,
    'nr_of_random_vectors_tried': 150,
    'max_nr_of_random_vectors_without_change': 100,
    'c_param_lower_factor': 0.2,
    'c_param_upper_factor': 2,
    'nr_of_c_params_tried_per_random_vector': 5,
    'nr_of_cluster_sizes_to_check': 15,
    'cluster_size_lower_factor': 0.4,
    'cluster_size_upper_factor': 1.5,
    'nr_of_ind_sets_to_find_in_multiple_sets_strategy': 4,
}


def better_ind_sets(graph, ind_sets1, ind_sets2):
    """Returns true if the first list of independent sets is better than the second."""

    if ind_sets2 is None or len(ind_sets2) == 0:
        return True

    if ind_sets1 is None or len(ind_sets1) == 0:
        return False

    temp_colors1 = {v: -1 for s in ind_sets1 for v in s}
    temp_colors2 = {v: -1 for s in ind_sets2 for v in s}

    if len(set(temp_colors2.values())) == 0:
        return False

    if len(set(temp_colors1.values())) == 0:
        return False

    color = 0
    for ind_set in ind_sets1:
        color += 1
        for v in ind_set:
            temp_colors1[v] = color

    color = 0
    for ind_set in ind_sets2:
        color += 1
        for v in ind_set:
            temp_colors2[v] = color

    avg1 = len(temp_colors1.keys()) / float(len(set(temp_colors1.values())))
    avg2 = len(temp_colors2.keys()) / float(len(set(temp_colors2.values())))

    # It is not optimal as there may be some set that is completely contained in sum of other sets and only
    #   by coloring it first, will we spare its color

    # set_sizes1 = map(lambda s: len(s), ind_sets1)
    # set_sizes2 = map(lambda s: len(s), ind_sets2)
    # avg1 = float(sum(set_sizes1)) / len(set_sizes1)
    # avg2 = float(sum(set_sizes2)) / len(set_sizes2)

    return avg1 > avg2


def color_by_independent_sets(graph, L, colors, init_params):
    """This strategy finds one or more independent set finding it one list of sets at a time."""

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

    # TODO: strategy of determining how many sets to get at once from find_ind_set_strategy

    best_ind_sets = None
    for it in range(params['nr_of_times_restarting_ind_set_strategy']):
        ind_sets = init_params['find_independent_sets_strategy'](
            graph, L, init_params, nr_of_sets=1)  # Returns list of sets

        if better_ind_sets(graph, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets

    update_colors_and_graph(graph, colors, best_ind_sets)
    logging.debug('Found independent sets (maybe identical) of sizes: ' + str([len(s) for s in best_ind_sets]))


def find_ind_set_by_random_vector_projection(graph, L, init_params, nr_of_sets=1):
    """KMS according to Arora, Chlamtac, Charikar.

    Tries to return nr_of_sets but might return less.
    """

    def better_subgraph(subg_v1, subg_e1, subg_v2, subg_e2):
        """Returns True if subg1 is a better subgraph than subg2. Subgraph1 is represented by its vertices and edges.
            subg_v1 and subg_e1."""

        if subg_v2 is None or len(subg_v2) == 0:
            return True

        if subg_v1 is None or len(subg_v1) == 0:
            return False

        ind_set1 = extract_independent_subset(
            subg_v1, subg_e1, strategy=init_params['independent_set_extraction_strategy'])
        ind_set2 = extract_independent_subset(
            subg_v2, subg_e2, strategy=init_params['independent_set_extraction_strategy'])

        return len(ind_set1) > len(ind_set2)

    def compute_c_opt(g, L):
        """Computes optimal c parameter according to KMS.

        Args:
            g (nx.Graph): Graph
            L (2-dim array): Vector coloring of graph
        """

        max_degree = max(dict(g.degree()).values())
        k = find_number_of_vector_colors_from_vector_coloring(g, L)
        temp = (2 * (k - 2) * math.log(max_degree)) / k
        if temp >= 0:
            c = math.sqrt(temp)
        else:
            c = 0.0

        return c

    c_opt = compute_c_opt(graph, L)

    n = graph.number_of_nodes()
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(graph.nodes()))}

    best_subgraph_edges = None
    best_subgraph_nodes = None
    it = 0
    last_change = 0
    while it < params['nr_of_random_vectors_tried'] \
            and it - last_change < params['max_nr_of_random_vectors_without_change']:
        it += 1
        r = np.random.normal(0, 1, n)
        x = np.dot(L, r)
        for c in np.linspace(
                c_opt * params['c_param_lower_factor'],
                c_opt * params['c_param_upper_factor'],
                num=params['nr_of_c_params_tried_per_random_vector']):
            current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c}
            current_subgraph_edges = {(i, j) for i, j in graph.edges() if
                                      (i in current_subgraph_nodes and j in current_subgraph_nodes)}

            if better_subgraph(current_subgraph_nodes, current_subgraph_edges, best_subgraph_nodes,
                               best_subgraph_edges):
                best_subgraph_nodes = current_subgraph_nodes
                best_subgraph_edges = current_subgraph_edges
                last_change = it

    return [extract_independent_subset(
        best_subgraph_nodes, best_subgraph_edges, strategy=init_params['independent_set_extraction_strategy'])]


def find_ind_set_by_clustering(graph, L, init_params, nr_of_sets=1):
    """Returns independent sets. Tries to return nr_of_sets but might return less."""

    z = linkage(L, method='complete', metric='cosine')

    k = find_number_of_vector_colors_from_vector_coloring(graph, L)
    opt_t = 1 + 1 / (k - 1) - 0.001 if k > 1.5 else 2.0  # Should guarantee each cluster can be colored with one color
    # t *= 1.1  # Make clusters a bit bigger

    best_ind_set = None
    for t in np.linspace(
            opt_t * params['cluster_size_lower_factor'],
            opt_t * params['cluster_size_upper_factor'],
            num=params['nr_of_cluster_sizes_to_check']):
        clusters = fcluster(z, t, criterion='distance')
        partition = {n: clusters[v] for v, n in enumerate(sorted(list(graph.nodes())))}

        # Find biggest cluster
        freq = {}
        for key, value in partition.items():
            if value not in freq:
                freq[value] = 1
            else:
                freq[value] += 1
        clst = max(freq, key=freq.get)

        vertices = {v for v, clr in partition.items() if clr == clst}
        edges = nx.subgraph(graph, vertices).edges()
        ind_set = [extract_independent_subset(
            vertices, edges, strategy=init_params['independent_set_extraction_strategy'])]

        if better_ind_sets(graph, ind_set, best_ind_set):
            best_ind_set = ind_set

    return best_ind_set
