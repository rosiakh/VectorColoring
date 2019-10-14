from solver.solver import compute_dummy_vector_coloring
from vector_projection_helper import *


def find_indsets_by_dummy_vector_strategy(graph, find_indsets_strategy_params, nr_of_sets, shmem_ind_sets=None,
                                          lock=None):
    """Dummy vector strategy.

    Tries to return nr_of_sets of independent sets but might return less.

    :param graph: (nx.Graph)
    :param find_indsets_strategy_params: parameters needed for dummy vector strategy
    :param nr_of_sets: [not used in strategy; reserved for future use]
    :param shmem_ind_sets: used only for parallel computing
    :param lock: used only for parallel computing
    :return (best_ind_sets, best_almost_ind_sets):
        best_ind_sets: list of independent sets of vertices
        best_almost_ind_sets: list of original sets of vertices that were used to extract the corresponding independent
            sets
    """

    dummy_vector_coloring = compute_dummy_vector_coloring(
        graph, find_indsets_strategy_params['beta_factor_strategy'],
        find_indsets_strategy_params['alpha_upper_bound'])

    best_ind_sets = []
    best_almost_ind_sets = []

    # this single loop serves only to make the function similar in structure to other strategies. it can also possibly
    # be used in future together with some strategy parameter
    for _ in range(1):
        if find_indsets_strategy_params['find_almost_indsets_strategy'] == "projection":
            ind_sets, almost_indsets = find_indsets_by_dummy_vector_projection(graph, find_indsets_strategy_params,
                                                                               dummy_vector_coloring, nr_of_sets)
        elif find_indsets_strategy_params['find_almost_indsets_strategy'] == "greedy":
            ind_sets, almost_indsets = find_indsets_by_dummy_vector_greedy(graph, find_indsets_strategy_params,
                                                                           dummy_vector_coloring, nr_of_sets)
        else:
            raise Exception('Unknown dummy vector projection find almost indsets strategy')

        if is_better_ind_sets(graph, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets
            best_almost_ind_sets = almost_indsets

    if shmem_ind_sets is not None and lock is not None:
        lock.acquire()
        shmem_ind_sets.append(best_ind_sets)
        lock.release()

    return best_ind_sets, best_almost_ind_sets


def find_indsets_by_dummy_vector_projection(graph, find_indsets_strategy_params, dummy_vector_coloring, nr_of_sets):
    """Find independent set using dummy vector projection strategy.

    :param graph: (nx.Graph)
    :param find_indsets_strategy_params: parameters needed for dummy vector projection strategy
    :param dummy_vector_coloring: (2-dim matrix (n+1)x(n+1)) vector coloring with one more dummy vector added as n+1-st
        vector
    :param nr_of_sets: [reserved for future use]
    :return: (ind_sets, almost_ind_sets):
        ind_sets: list of independent sets of vertices (list of size 1)
        almost_ind_sets: list of sets of vertices that served as a basis for extraction of corresponding independent
            sets in ind_sets (list of size 1)
    """

    c_params = find_indsets_strategy_params['c_adaptation_strategy_params']
    c_opt = compute_dummy_c_opt(graph, dummy_vector_coloring, c_params['initial_c_percentile'])

    ind_sets = []
    almost_ind_sets = []
    n = graph.number_of_nodes()
    r = -dummy_vector_coloring[n]
    best_ind_set, best_almost_ind_set = obtain_single_ind_set_by_projection(
        dummy_vector_coloring, r, find_indsets_strategy_params, c_opt, graph)

    ind_sets.extend(best_ind_set)
    almost_ind_sets.extend(best_almost_ind_set)

    return ind_sets, almost_ind_sets


def find_indsets_by_dummy_vector_greedy(graph, find_indsets_strategy_params, dummy_vector_coloring, nr_of_sets):
    """Find

    :param graph: (nx.Graph)
    :param find_indsets_strategy_params: parameters needed for dummy vector greedy strategy
    :param dummy_vector_coloring: (2-dim matrix (n+1)x(n+1)) vector coloring with one more dummy vector added as n+1-st
        vector
    :param nr_of_sets: [reserved for future use]
    :return:
    """

    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(graph.nodes()))}
    n = graph.number_of_nodes()
    r = -dummy_vector_coloring[n]

    from scipy.spatial.distance import cdist
    distances = cdist([-r], dummy_vector_coloring[0:n], 'cosine')

    ind_set = []
    sorted_distances = np.argsort(distances)[0]
    i = 0
    has_edge_flag = False
    while i < len(sorted_distances) and is_continue_greedy(has_edge_flag, find_indsets_strategy_params, ind_set,
                                                           sorted_distances, i):
        w = sorted_distances[i]
        has_edge_flag = False
        for v in ind_set:
            if has_edge_between_ith_and_jth(graph, w, v):
                has_edge_flag = True
                break
        if not has_edge_flag:
            ind_set.append(w)
        i += 1

    ind_set_vertices = [inv_vertices_mapping[j] for j in ind_set]
    return [ind_set_vertices], [[v for j, v in enumerate(sorted_distances) if j < i]]


def is_continue_greedy(has_edge_flag, find_indsets_strategy_params, ind_set, sorted_distances, i):
    """
    :param has_edge_flag:
    :param find_indsets_strategy_params:
    :param ind_set:
    :param sorted_distances:
    :param i:
    :return:
    """
    if find_indsets_strategy_params['greedy_continue_strategy'] == 'all-vertices':
        return True
    elif find_indsets_strategy_params['greedy_continue_strategy'] == 'first-edge':
        return not has_edge_flag
    elif find_indsets_strategy_params['greedy_continue_strategy'] == 'ratio':
        greedy_params = find_indsets_strategy_params['greedy_continue_strategy_params']
        if i < greedy_params['lower_bound_nr_of_nodes']:
            return True
        ratio = float(len(ind_set)) / float(i)
        return ratio > greedy_params['ratio_upper_bound']


def compute_dummy_c_opt(graph, dummy_vector_coloring, percentile):
    """
    :param graph:
    :param dummy_vector_coloring:
    :param percentile:
    :return:
    """
    dummy_matrix_coloring = np.dot(dummy_vector_coloring, np.transpose(dummy_vector_coloring))
    n = dummy_matrix_coloring.shape[0]
    dummy_dot_products = dummy_matrix_coloring[n - 1][0:n - 1]

    # draw_distributions(dummy_matrix_coloring, 5.0)
    return -np.percentile(dummy_dot_products, percentile)
