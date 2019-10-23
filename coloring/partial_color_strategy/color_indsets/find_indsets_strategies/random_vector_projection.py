import math

from solver.solver import compute_vector_coloring
from vector_projection_helper import *


def find_indsets_by_random_vector_projection_strategy(graph, find_indsets_strategy_params, nr_of_sets,
                                                      shmem_ind_sets=None, lock=None):
    """KMS according to Arora, Chlamtac, Charikar.

    While number of iterations and number of iterations without improvement do not exceed their respective thresholds,
    the function asks find_indsets_by_random_vector_projection function for list of independent sets and ultimately
    returns the best list and the list of almost independent sets from which they were extracted. It tries to return
    'nr_of_sets' independent sets in resulting list but does not guarantee that.

    :param graph: (nx.Graph)
    :param find_indsets_strategy_params: parameters for this and lower-level function
    :param nr_of_sets: number of independent sets that it tries to obtain
    :param shmem_ind_sets: used for parallel computation only
    :param lock: used for parallel computation only
    :return two lists of sets :
        ind_sets : list of size 'nr_of_sets' of independent sets
        almost_indsets : list of size 'nr_of_sets' of possibly not independent sets that were used to create ind_sets
            (set nr x in this list was used to create set nr x in ind_sets)
    """

    vector_coloring = compute_vector_coloring(graph, find_indsets_strategy_params['sdp_type'])

    best_ind_sets = []
    best_almost_indsets = []
    it = 0
    last_change = 0
    while it < find_indsets_strategy_params['nr_of_random_vector_sets_to_try'] \
            and it - last_change < find_indsets_strategy_params['max_nr_of_random_vectors_without_change']:
        it += 1

        ind_sets, almost_indsets = find_indsets_by_random_vector_projections(graph, find_indsets_strategy_params,
                                                                             vector_coloring, nr_of_sets)

        if is_better_ind_sets(graph, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets
            best_almost_indsets = almost_indsets
            last_change = it

    if shmem_ind_sets is not None and lock is not None:
        lock.acquire()
        shmem_ind_sets.append(best_ind_sets)
        lock.release()

    return best_ind_sets, best_almost_indsets


def find_indsets_by_random_vector_projections(graph, find_indsets_strategy_params, vector_coloring, nr_of_sets):
    """Find list of independent sets by taking 'nr_of_sets' random pivot vectors and using projection to obtain one independent
        set for each pivot vector. Different pivot vectors might produce identical independent sets.

    :param graph: (nx.Graph)
    :param find_indsets_strategy_params: parameters for lower-level function
    :param vector_coloring: standard vector coloring of graph
    :param nr_of_sets: number of random pivot vectors and subsequently number of independent sets returned (they might
        be identical)
    :return: returns lists of sets :
        ind_sets : list of size 'nr_of_sets' of independent sets
        almost_indsets : list of size 'nr_of_sets' of possibly not independent sets that were used to create ind_sets
            (set nr x in this list was used to create set nr x in ind_sets)
    """

    c_opt = compute_c_opt(graph, vector_coloring)
    n = graph.number_of_nodes()

    ind_sets = []
    almost_ind_sets = []
    for _ in range(nr_of_sets):
        pivot_vector = np.random.normal(0, 1, n)
        best_ind_set, best_almost_ind_set = obtain_single_ind_set_by_projection(vector_coloring, pivot_vector,
                                                                                find_indsets_strategy_params, c_opt,
                                                                                graph)

        ind_sets.extend(best_ind_set)
        almost_ind_sets.extend(best_almost_ind_set)

    return ind_sets, almost_ind_sets


def compute_c_opt(graph, vector_coloring, is_r_normalized=False):
    """Computes optimal c parameter according to KMS. The parameter is used as a threshold of value of projection of
        given vertex-vector on a pivot vector, above which vertex-vectors are taken into almost independent set
        associated with pivot vector.

    :param graph: (nx.Graph)
    :param vector_coloring: (n x n matrix) standard vector coloring of graph
    :param is_r_normalized: (bool) always False
    :return c parameter computed according to modified method from KMS
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
