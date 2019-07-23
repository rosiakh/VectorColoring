import math

import networkx as nx

from coloring.algorithm_helper import *
from coloring.partial_color_strategy.color_indsets.color_indsets_helper import *
from solver.solver import compute_vector_coloring


def find_indsets_by_random_vector_projection_strategy(graph, find_indsets_strategy_params, nr_of_sets,
                                                      shmem_ind_sets=None, lock=None):
    """KMS according to Arora, Chlamtac, Charikar.

    Tries to return nr_of_sets of independent sets but might return less.
    """

    vector_coloring = compute_vector_coloring(graph, find_indsets_strategy_params['sdp_type'])

    best_ind_sets = []
    it = 0
    last_change = 0
    while it < find_indsets_strategy_params['nr_of_random_vector_sets_to_try'] \
            and it - last_change < find_indsets_strategy_params['max_nr_of_random_vectors_without_change']:
        it += 1

        ind_sets, is_change = find_indsets_by_random_vector_projections(graph, find_indsets_strategy_params,
                                                                        vector_coloring, nr_of_sets)
        if is_change:
            last_change = it

        if is_better_ind_sets(graph, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets

    if shmem_ind_sets is not None and lock is not None:
        lock.acquire()
        shmem_ind_sets.append(best_ind_sets)
        lock.release()

    return best_ind_sets


def find_indsets_by_random_vector_projections(graph, find_indsets_strategy_params, vector_coloring, nr_of_sets):
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(graph.nodes()))}
    c_opt = compute_c_opt(graph, vector_coloring)
    n = graph.number_of_nodes()

    ind_sets = []
    is_change = False
    for i in range(nr_of_sets):
        r = np.random.normal(0, 1, n)
        x = np.dot(vector_coloring, r)
        best_ind_set = []
        for c in np.linspace(
                c_opt * find_indsets_strategy_params['c_param_lower_factor'],
                c_opt * find_indsets_strategy_params['c_param_upper_factor'],
                num=find_indsets_strategy_params[
                    'nr_of_c_params_tried_per_random_vector']):
            current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c}
            current_subgraph_edges = {(i, j) for i, j in graph.edges() if
                                      (i in current_subgraph_nodes and j in current_subgraph_nodes)}

            ind_set = [extract_independent_subset(
                current_subgraph_nodes, current_subgraph_edges,
                strategy=find_indsets_strategy_params['independent_set_extraction_strategy'])]

            if is_better_ind_sets(graph, ind_set, best_ind_set):
                best_ind_set = ind_set
                is_change = True

        ind_sets.extend(best_ind_set)

    return ind_sets, is_change


def compute_c_opt(graph, vector_coloring):
    """Computes optimal c parameter according to KMS.

    Args:
        graph (nx.Graph): Graph
        vector_coloring (2-dim array): Vector coloring of graph
    """

    max_degree = max(dict(graph.degree()).values())
    k = find_number_of_vector_colors_from_vector_coloring(graph, vector_coloring)
    temp = (2 * (k - 2) * math.log(max_degree)) / k
    if temp >= 0:
        c = math.sqrt(temp)
    else:
        c = 0.0

    return c
