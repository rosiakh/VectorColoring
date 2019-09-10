from coloring.algorithm_helper import *
from coloring.partial_color_strategy.color_indsets.color_indsets_helper import *


def find_ind_set_for_c(graph, x, c, inv_vertices_mapping, find_indsets_strategy_params):
    current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c}
    current_subgraph_edges = {(i, j) for i, j in graph.edges() if
                              (i in current_subgraph_nodes and j in current_subgraph_nodes)}

    ind_set = [extract_independent_subset(
        current_subgraph_nodes, current_subgraph_edges,
        strategy=find_indsets_strategy_params['independent_set_extraction_strategy'])]

    return ind_set, current_subgraph_nodes


def obtain_single_ind_set_by_projection(vector_coloring, r, find_indsets_strategy_params, c_opt, graph):
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(graph.nodes()))}
    c_params = find_indsets_strategy_params['c_adaptation_strategy_params']
    n = graph.number_of_nodes()
    x = np.dot(vector_coloring, r)
    best_ind_set = []
    best_almost_ind_set = []

    if find_indsets_strategy_params['c_adaptation_strategy'] == 'linspace':
        for c in np.linspace(
                c_opt * c_params['c_param_lower_factor'],
                c_opt * c_params['c_param_upper_factor'],
                num=c_params['nr_of_c_params_tried_per_random_vector']):

            ind_set, current_subgraph_nodes = \
                find_ind_set_for_c(graph, x, c, inv_vertices_mapping, find_indsets_strategy_params)

            if is_better_ind_sets(graph, ind_set, best_ind_set):
                best_ind_set = ind_set
                best_almost_ind_set = [current_subgraph_nodes]
                is_change = True

    elif find_indsets_strategy_params['c_adaptation_strategy'] == 'ratio':
        c = c_opt
        while True:

            ind_set, current_subgraph_nodes = \
                find_ind_set_for_c(graph, x, c, inv_vertices_mapping, find_indsets_strategy_params)

            if is_better_ind_sets(graph, ind_set, best_ind_set):
                best_ind_set = ind_set
                best_almost_ind_set = [current_subgraph_nodes]
                is_change = True

            current_ratio = float(len(ind_set[0])) / float(len(current_subgraph_nodes))
            if c < c_params['c_lower_bound'] or current_ratio < c_params['ratio_upper_bound'] \
                    or len(current_subgraph_nodes) == n:
                break
            else:
                c *= c_params['c_decrease_ratio']

    else:
        raise Exception("Unknown c adaptation strategy")

    return best_ind_set, best_almost_ind_set
