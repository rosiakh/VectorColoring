from coloring.algorithm_helper import *
from coloring.partial_color_strategy.color_indsets.color_indsets_helper import *
from solver.solver import compute_dummy_vector_coloring


def find_indsets_by_dummy_vector_strategy(graph, find_indsets_strategy_params, nr_of_sets,
                                          shmem_ind_sets=None, lock=None):
    """ Dummy vector projection.

    Tries to return nr_of_sets of independent sets but might return less.
    """

    dummy_vector_coloring = compute_dummy_vector_coloring(
        graph, find_indsets_strategy_params['beta_factor_strategy'],
        find_indsets_strategy_params['is_alpha_constrained'])

    best_ind_sets = []

    if find_indsets_strategy_params['find_almost_indsets_strategy'] == "projection":
        ind_sets = find_indsets_by_dummy_vector_projection(graph, find_indsets_strategy_params,
                                                           dummy_vector_coloring, nr_of_sets)
    elif find_indsets_strategy_params['find_almost_indsets_strategy'] == "greedy":
        ind_sets = find_indsets_by_dummy_vector_greedy(graph, find_indsets_strategy_params,
                                                       dummy_vector_coloring, nr_of_sets)
    else:
        raise Exception('Unknown dummy vector projection find almost indsets strategy')

    if is_better_ind_sets(graph, ind_sets, best_ind_sets):
        best_ind_sets = ind_sets

    if shmem_ind_sets is not None and lock is not None:
        lock.acquire()
        shmem_ind_sets.append(best_ind_sets)
        lock.release()

    return best_ind_sets


def find_indsets_by_dummy_vector_projection(graph, find_indsets_strategy_params, dummy_vector_coloring, nr_of_sets):
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(graph.nodes()))}
    n = graph.number_of_nodes()
    c_opt = compute_dummy_c_opt(graph, dummy_vector_coloring)

    ind_sets = []
    r = -dummy_vector_coloring[n]
    x = np.dot(dummy_vector_coloring, r)
    best_ind_set = []
    for c in np.linspace(
            c_opt * find_indsets_strategy_params['c_param_lower_factor'],
            c_opt * find_indsets_strategy_params['c_param_upper_factor'],
            num=find_indsets_strategy_params['nr_of_c_params_tried_per_random_vector']):
        current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c}
        current_subgraph_edges = {(i, j) for i, j in graph.edges() if
                                  (i in current_subgraph_nodes and j in current_subgraph_nodes)}

        ind_set = [extract_independent_subset(
            current_subgraph_nodes, current_subgraph_edges,
            strategy=find_indsets_strategy_params['independent_set_extraction_strategy'])]

        if is_better_ind_sets(graph, ind_set, best_ind_set):
            best_ind_set = ind_set

    ind_sets.extend(best_ind_set)

    return ind_sets


def find_indsets_by_dummy_vector_greedy(graph, find_indsets_strategy_params, dummy_vector_coloring, nr_of_sets):
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(graph.nodes()))}
    n = graph.number_of_nodes()
    r = -dummy_vector_coloring[n]

    from scipy.spatial.distance import cdist
    distances = cdist([r], dummy_vector_coloring[0:n], 'cosine')

    ind_set = []
    sorted_distances = np.argsort(distances)[0]
    for i in sorted_distances:
        has_edge_flag = False
        for v in ind_set:
            if has_edge_between_ith_and_jth(graph, i, v):
                has_edge_flag = True
                break
        if not has_edge_flag:
            ind_set.append(i)

    ind_set_vertices = [inv_vertices_mapping[i] for i in ind_set]
    return [ind_set_vertices]


def compute_dummy_c_opt(graph, dummy_vector_coloring):
    dummy_matrix_coloring = np.dot(dummy_vector_coloring, np.transpose(dummy_vector_coloring))
    n = dummy_matrix_coloring.shape[0]
    dummy_dot_products = dummy_matrix_coloring[n - 1][0:n - 1]

    # draw_distributions(dummy_matrix_coloring, 5.0)
    return np.percentile(dummy_dot_products, 50)
