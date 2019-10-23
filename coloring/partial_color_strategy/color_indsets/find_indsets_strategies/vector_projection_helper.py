from coloring.algorithm_helper import *
from coloring.partial_color_strategy.color_indsets.color_indsets_helper import *


def find_ind_set_for_c(graph, pivot_dot_products, c, inv_vertices_mapping, find_indsets_strategy_params):
    """Obtains single independent set finding vertex-vectors whose dot product with pivot vector is above given threshold.

    The values of dot products are already given. This function chooses the vertices that are above threshold and creates
    almost independet set from them from which it later extracts independent set.

    :param graph: (nx.Graph)
    :param pivot_dot_products: list of values of dot product of vertex-vectors with some pivot vector
    :param c: threshold of dot product, above which we take given vertex-vector to almost independent set associated
        with pivot vector
    :param inv_vertices_mapping: inverted mapping of vertices names and ordinal numbers
    :param find_indsets_strategy_params: parameters for lower-level function
    :return
        ind_set : independent set of vertices extracted from 'current_subgraph_nodes'
        current_subgraph_nodes : vertices that are close enough to pivot vector that their projection on pivot is larger
            than threshold 'c'
    """

    current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(pivot_dot_products) if v >= c}
    current_subgraph_edges = {(i, j) for i, j in graph.edges() if
                              (i in current_subgraph_nodes and j in current_subgraph_nodes)}

    ind_set = [extract_independent_subset(
        current_subgraph_nodes, current_subgraph_edges,
        strategy=find_indsets_strategy_params['independent_set_extraction_strategy'])]

    return ind_set, current_subgraph_nodes


def obtain_single_ind_set_by_projection(vector_coloring, pivot_vector, find_indsets_strategy_params, c_opt, graph):
    """Given vector coloring (standard or dummy), a pivot vector, and theoretically good threshold 'c_opt' for value
    of dot product it returns single independent set.

    The independent set is obtained by obtaining and comparing a few different independent sets found using different
    values of dot product threshold. The thresholds used for finding those different independent set for comparison
    are computed using different 'c_adaptation_strategy' and are usually somehow based on 'c_opt'.

    :param vector_coloring: standard or dummy vector coloring
    :param pivot_vector: pivot vector used for projections
    :param find_indsets_strategy_params: parameters used here and in lower-level functions
    :param c_opt: base threshold for dot products with pivot_vector that is theoretically optimal
    :param graph: (nx.Graph)
    :return:
        best_ind_set : best independent set found
        best_almost_ind_set : almost independent set from which 'best_ind_set' was extracted
    """

    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(graph.nodes()))}
    c_params = find_indsets_strategy_params['c_adaptation_strategy_params']
    n = graph.number_of_nodes()
    x = np.dot(vector_coloring, pivot_vector)
    best_ind_set = []
    best_almost_ind_set = []

    if find_indsets_strategy_params['c_adaptation_strategy'] == 'linspace':
        # TODO: opisz te strategie
        for c in np.linspace(
                c_opt * c_params['c_param_lower_factor'],
                c_opt * c_params['c_param_upper_factor'],
                num=c_params['nr_of_c_params_tried_per_random_vector']):

            ind_set, current_subgraph_nodes = \
                find_ind_set_for_c(graph, x[:graph.number_of_nodes()], c, inv_vertices_mapping,
                                   find_indsets_strategy_params)

            if is_better_ind_sets(graph, ind_set, best_ind_set):
                best_ind_set = ind_set
                best_almost_ind_set = [current_subgraph_nodes]
                is_change = True

    elif find_indsets_strategy_params['c_adaptation_strategy'] == 'ratio':
        #TODO: opisz te strategie
        c = c_opt
        while True:
            ind_set, current_subgraph_nodes = find_ind_set_for_c(graph, x, c, inv_vertices_mapping,
                                                                 find_indsets_strategy_params)

            if is_better_ind_sets(graph, ind_set, best_ind_set):
                best_ind_set = ind_set
                best_almost_ind_set = [current_subgraph_nodes]
                is_change = True

            current_ratio = float(len(ind_set[0])) / float(len(current_subgraph_nodes)) if \
                len(current_subgraph_nodes) > 0 else 1
            if c < c_params['c_lower_bound'] or current_ratio < c_params['ratio_upper_bound'] \
                    or len(current_subgraph_nodes) == n:
                break
            else:
                c *= c_params['c_decrease_ratio']

    else:
        raise Exception("Unknown c adaptation strategy")

    return best_ind_set, best_almost_ind_set
