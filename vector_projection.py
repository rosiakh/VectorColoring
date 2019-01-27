import logging
import math

import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster

from algorithm_helper import *


def better_ind_sets(g, ind_sets1, ind_sets2):
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


def partially_color_graph_by_vector_projections(g, L, colors, find_ind_sets_strategy):
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

    # Config
    iterations = 10

    best_ind_sets = None
    for it in range(iterations):
        ind_sets = find_ind_sets_strategy(g, L)  # Returns list of sets

        if better_ind_sets(g, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets

    update_colors_and_graph(g, colors, best_ind_sets)
    logging.debug('Found independent sets (maybe identical) of sizes: ' + str([len(s) for s in best_ind_sets]))


def find_ind_set_by_random_vector_projection_kms(g, L):
    """KMS according to Arora, Chlamtac, Charikar"""

    def better_subgraph(subg_v1, subg_e1, subg_v2, subg_e2):
        """Returns True if subg1 is a better subgraph than subg2. Subgraph1 is represented by its vertices and edges.
            subg_v1 and subg_e1."""

        if subg_v2 is None or len(subg_v2) == 0:
            return True

        if subg_v1 is None or len(subg_v1) == 0:
            return False

        ind_set1 = extract_independent_subset(subg_v1, subg_e1, strategy='arora_kms')
        ind_set2 = extract_independent_subset(subg_v2, subg_e2, strategy='arora_kms')

        return len(ind_set1) > len(ind_set2)

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
    iterations = 1000
    iterations_wo_change = 100
    c = compute_c_opt(g, L)

    n = g.number_of_nodes()
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(g.nodes()))}

    best_subgraph_edges = None
    best_subgraph_nodes = None
    it = 0
    last_change = 0
    while it < iterations and it - last_change < iterations_wo_change:
        it += 1
        r = np.random.normal(0, 1, n)
        x = np.dot(L, r)
        current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c}
        current_subgraph_edges = {(i, j) for i, j in g.edges() if
                                  (i in current_subgraph_nodes and j in current_subgraph_nodes)}

        if better_subgraph(current_subgraph_nodes, current_subgraph_edges, best_subgraph_nodes, best_subgraph_edges):
            best_subgraph_nodes = current_subgraph_nodes
            best_subgraph_edges = current_subgraph_edges
            last_change = it

    return [extract_independent_subset(best_subgraph_nodes, best_subgraph_edges, strategy='arora_kms')]


def find_ind_set_by_random_vector_projection_kms_prim(g, L):
    """KMS according to Arora, Chlamtac, Charikar"""

    def better_subgraph(subg_v1, subg_e1, subg_v2, subg_e2):
        """Returns True if subg1 is a better subgraph than subg2. Subgraph1 is represented by its vertices and edges.
            subg_v1 and subg_e1."""

        if subg_v2 is None or len(subg_v2) == 0:
            return True

        if subg_v1 is None or len(subg_v1) == 0:
            return False

        ind_set1 = extract_independent_subset(subg_v1, subg_e1, strategy='arora_kms_prim')
        ind_set2 = extract_independent_subset(subg_v2, subg_e2, strategy='arora_kms_prim')

        return len(ind_set1) > len(ind_set2)

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
    iterations = 10
    iterations_wo_change = 100
    # c = compute_c_opt(g, L)

    n = g.number_of_nodes()
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(g.nodes()))}

    best_subgraph_edges = None
    best_subgraph_nodes = None
    it = 0
    last_change = 0
    while it < iterations and it - last_change < iterations_wo_change:
        it += 1
        r = np.random.normal(0, 1, n)
        x = np.dot(L, r)
        # logging.debug('min(x): {1}    max(x): {0}'.format(max(x), min(x)))
        for c in np.linspace(0, max(x), num=8):
            current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c}
            current_subgraph_edges = {(i, j) for i, j in g.edges() if
                                      (i in current_subgraph_nodes and j in current_subgraph_nodes)}

            if better_subgraph(current_subgraph_nodes, current_subgraph_edges, best_subgraph_nodes,
                               best_subgraph_edges):
                best_subgraph_nodes = current_subgraph_nodes
                best_subgraph_edges = current_subgraph_edges
                last_change = it

    return [extract_independent_subset(best_subgraph_nodes, best_subgraph_edges, strategy='arora_kms_prim')]




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

        ind_set1 = extract_independent_subset(subg_v1, subg_e1, strategy='max_degree_first')
        ind_set2 = extract_independent_subset(subg_v2, subg_e2, strategy='max_degree_first')

        return len(ind_set1) > len(ind_set2)

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
    iterations = 1000
    iterations_wo_change = 100
    c = 0.7 * compute_c_opt(g, L)

    n = g.number_of_nodes()
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(g.nodes()))}

    best_subgraph_edges = None
    best_subgraph_nodes = None
    it = 0
    last_change = 0
    while it < iterations and it - last_change < iterations_wo_change:
        it += 1
        r = np.random.normal(0, 1, n)
        x = np.dot(L, r)
        current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c}
        current_subgraph_edges = {(i, j) for i, j in g.edges() if
                                  (i in current_subgraph_nodes and j in current_subgraph_nodes)}

        if better_subgraph(current_subgraph_nodes, current_subgraph_edges, best_subgraph_nodes, best_subgraph_edges):
            best_subgraph_nodes = current_subgraph_nodes
            best_subgraph_edges = current_subgraph_edges
            last_change = it

    return [extract_independent_subset(best_subgraph_nodes, best_subgraph_edges)]


def find_multiple_ind_sets_by_random_vector_projections(g, L):
    iterations = 100
    nr_of_sets = 4

    it = 0
    ind_sets = []
    while len(ind_sets) < nr_of_sets and it < iterations:
        ind_sets.extend(find_ind_set_by_random_vector_projection(g, L))

    return ind_sets


def find_ind_set_by_clustering(g, L):
    Z = linkage(L, method='complete', metric='cosine')

    # show_dendrogram(Z)

    k = find_number_of_vector_colors_from_vector_coloring(g, L)
    opt_t = 1 + 1 / (
            k - 1) - 0.001 if k > 1.5 else 2.0  # Should guarantee that each cluster can be colored with one color
    # t *= 1.1  # Make clusters a bit bigger

    best_ind_set = None
    for t in np.linspace(opt_t - 0.3, opt_t + 0.3, 20):
        clusters = fcluster(Z, t, criterion='distance')
        partition = {n: clusters[v] for v, n in enumerate(sorted(list(g.nodes())))}

        # Find biggest cluster
        freq = {}
        for key, value in partition.items():
            if value not in freq:
                freq[value] = 1
            else:
                freq[value] += 1
        clst = max(freq, key=freq.get)

        vertices = {v for v, clr in partition.items() if clr == clst}
        edges = nx.subgraph(g, vertices).edges()
        ind_set = [extract_independent_subset(vertices, edges)]

        if better_ind_sets(g, ind_set, best_ind_set):
            best_ind_set = ind_set

    return best_ind_set
