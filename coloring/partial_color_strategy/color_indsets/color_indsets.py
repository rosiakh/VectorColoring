import logging

from coloring.partial_color_strategy.color_indsets.color_indsets_helper import *
from coloring.partial_color_strategy.color_indsets.find_indsets_strategies.clustering import find_ind_sets_by_clustering
from coloring.partial_color_strategy.color_indsets.find_indsets_strategies.dummy_vector_strategy import \
    find_indsets_by_dummy_vector_strategy
from coloring.partial_color_strategy.color_indsets.find_indsets_strategies.random_vector_projection import \
    find_indsets_by_random_vector_projection_strategy

# from coloring.partial_color_strategy.color_indsets.find_indsets_strategies.dummy_vector_projection import

find_independent_sets_strategy_map = {
    'random_vector_projection': find_indsets_by_random_vector_projection_strategy,
    'clustering': find_ind_sets_by_clustering,
    'dummy_vector_strategy': find_indsets_by_dummy_vector_strategy,
    None: None,
}


def color_by_independent_sets(graph, partial_coloring, partial_color_strategy_params):
    """This strategy finds one or more independent set finding it one list of sets at a time."""

    logging.info('Looking for independent sets...')

    # TODO: strategy of determining how many sets to get at once from find_ind_set_strategy

    best_ind_sets = None
    best_almost_indsets = None
    nr_of_trials = 1 if partial_color_strategy_params['deterministic'] else \
        partial_color_strategy_params['nr_of_times_restarting_ind_set_strategy']

    for _ in range(nr_of_trials):
        ind_sets, almost_indsets = find_independent_sets_strategy_map[
            partial_color_strategy_params['find_independent_sets_strategy']](
            graph,
            partial_color_strategy_params['find_indsets_strategy_params'],
            nr_of_sets=get_nr_of_sets_at_once(graph))  # Returns list of sets

        if is_better_ind_sets(graph, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets
            best_almost_indsets = almost_indsets

    ratios = [str(len(x)) + "/" + str(len(y)) for x, y in zip(best_ind_sets, best_almost_indsets)]
    logging.info('Found independent sets (maybe identical) of sizes: ' + str(ratios))
    nr_of_colored_vertices = update_coloring_and_graph(graph, partial_coloring, best_ind_sets)
    logging.info('Nr of actually colored vertices: ' + str(nr_of_colored_vertices))


def update_coloring_and_graph(graph, partial_coloring, ind_sets):
    nr_of_colored_vertices = 0
    color = max(partial_coloring.values())
    for ind_set in ind_sets:
        color += 1
        for v in ind_set:
            if partial_coloring[v] == -1:
                partial_coloring[v] = color
                nr_of_colored_vertices += 1
        graph.remove_nodes_from(ind_set)

    return nr_of_colored_vertices


def get_nr_of_sets_at_once(graph):
    """Determines maximal number of independent sets found for one vector coloring."""

    # if nx.classes.density(graph) > 0.9 and graph.number_of_nodes() > 100:
    #     return max(1, int(math.floor((nx.classes.density(graph) + 0.5) * (graph.number_of_nodes() - 50) / 25)))
    #
    # if graph.number_of_nodes() > 130:
    #     return 5

    return 1
