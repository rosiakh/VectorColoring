import math

import networkx as nx

from solver.solver import compute_vector_coloring
from vector_projection_helper import *


def find_indsets_by_random_vector_projection_strategy(graph, find_indsets_strategy_params, nr_of_sets,
                                                      shmem_ind_sets=None, lock=None):
    """KMS according to Arora, Chlamtac, Charikar.

    Tries to return nr_of_sets of independent sets but might return less.
    """

    vector_coloring = compute_vector_coloring(graph, find_indsets_strategy_params['sdp_type'])

    best_ind_sets = []
    best_almost_indsets = []
    it = 0
    last_change = 0
    while it < find_indsets_strategy_params['nr_of_random_vector_sets_to_try'] \
            and it - last_change < find_indsets_strategy_params['max_nr_of_random_vectors_without_change']:
        it += 1

        ind_sets, almost_indsets, is_change = find_indsets_by_random_vector_projections(graph,
                                                                                        find_indsets_strategy_params,
                                                                                        vector_coloring, nr_of_sets)
        if is_change:
            last_change = it

        if is_better_ind_sets(graph, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets
            best_almost_indsets = almost_indsets

    if shmem_ind_sets is not None and lock is not None:
        lock.acquire()
        shmem_ind_sets.append(best_ind_sets)
        lock.release()

    return best_ind_sets, best_almost_indsets


def find_indsets_by_random_vector_projections(graph, find_indsets_strategy_params, vector_coloring, nr_of_sets):
    c_opt = compute_c_opt(graph, vector_coloring, is_r_normalized=False)
    n = graph.number_of_nodes()

    ind_sets = []
    almost_ind_sets = []
    is_change = False
    for _ in range(nr_of_sets):
        r = np.random.normal(0, 1, n)
        best_ind_set, best_almost_ind_set = obtain_single_ind_set_by_projection(
            vector_coloring, r, find_indsets_strategy_params, c_opt, graph)

        ind_sets.extend(best_ind_set)
        almost_ind_sets.extend(best_almost_ind_set)

    return ind_sets, almost_ind_sets, is_change


def compute_c_opt(graph, vector_coloring, is_r_normalized):
    """Computes optimal c parameter according to KMS.

    Args:
        graph (nx.Graph): Graph
        vector_coloring (2-dim array): Vector coloring of graph
    """
    c = 0.0
    if not is_r_normalized:
        max_degree = max(dict(graph.degree).values())
        k = find_number_of_vector_colors_from_vector_coloring(graph, vector_coloring)
        temp = (2 * (k - 2) * math.log(max_degree)) / (k ** 2)
        if temp >= 0:
            c = math.sqrt(temp)
        else:
            c = 0.0

    return c
