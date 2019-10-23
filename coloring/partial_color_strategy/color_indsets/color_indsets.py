import logging

from coloring.partial_color_strategy.color_indsets.color_indsets_helper import *
from coloring.partial_color_strategy.color_indsets.find_indsets_strategies.clustering import find_ind_sets_by_clustering
from coloring.partial_color_strategy.color_indsets.find_indsets_strategies.dummy_vector_strategy import \
    find_indsets_by_dummy_vector_strategy
from coloring.partial_color_strategy.color_indsets.find_indsets_strategies.random_vector_projection import \
    find_indsets_by_random_vector_projection_strategy

find_independent_sets_strategy_map = {
    'random_vector_projection': find_indsets_by_random_vector_projection_strategy,
    'clustering': find_ind_sets_by_clustering,
    'dummy_vector_strategy': find_indsets_by_dummy_vector_strategy,
    None: None,
}


def color_by_independent_sets(graph, partial_coloring, partial_color_strategy_params):
    """This strategy finds one or more independent set finding it one list of sets at a time.

    If the partial color strategy is deterministic, this function uses find_independet_sets_strategy to find a list
    of independent sets of graph. It asks for some number of independent sets (the number is computed in get_nr_of_sets_at_once
    function). After receiving list of independent sets it colors each one with different color.

    :param graph: (nx.Graph)
    :param partial_coloring: current legal (neighboring vertices have different colors or no color) partial coloring of graph
    :param partial_color_strategy_params: set of parameters
    """

    logging.info('Looking for independent sets...')

    best_ind_sets = None
    best_almost_indsets = None
    nr_of_trials = 1 if partial_color_strategy_params['deterministic'] else \
        partial_color_strategy_params['nr_of_times_restarting_ind_set_strategy']

    for _ in range(nr_of_trials):
        # Returns lists of sets
        # ind_sets : list of size 'nr_of_sets' of independent sets
        # almost_indsets : list of size 'nr_of_sets' of possibly not independent sets that were used to create ind_sets
        #   (set nr x in this list was used to create set nr x in ind_sets)
        ind_sets, almost_indsets = find_independent_sets_strategy_map[
            partial_color_strategy_params['find_independent_sets_strategy']](
            graph,
            partial_color_strategy_params['find_indsets_strategy_params'],
            nr_of_sets=get_nr_of_sets_at_once(graph))

        if is_better_ind_sets(graph, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets
            best_almost_indsets = almost_indsets

    # compute ratio of sizes of independent sets and almost independent sets from which they were extracted
    ratios = [str(len(x)) + "/" + str(len(y)) for x, y in zip(best_ind_sets, best_almost_indsets)]
    logging.info('Found independent sets (maybe identical) of sizes: ' + str(ratios))
    nr_of_colored_vertices = update_coloring_and_graph(graph, partial_coloring, best_ind_sets)
    logging.info('Nr of actually colored vertices: ' + str(nr_of_colored_vertices))


def update_coloring_and_graph(graph, partial_coloring, ind_sets):
    """Color independent sets.

    :param graph: (nx.Graph)
    :param partial_coloring: current legal (neighboring vertices have different colors or no color) partial coloring of graph
    :param ind_sets: list of independent sets that are to be colored with different colors
    :return: number of vertices that were actually colored
    """
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
    """Determines number of independent sets that we wish to receive from find_independent_sets_strategy
        (we might actually receive less)

    :param graph: (nx.Graph)
    :return number of independent sets that we wish to receive from find_independent_sets_strategy (we might actually
        receive less)
    """

    # if nx.classes.density(graph) > 0.9 and graph.number_of_nodes() > 100:
    #     return max(1, int(math.floor((nx.classes.density(graph) + 0.5) * (graph.number_of_nodes() - 50) / 25)))
    #
    # if graph.number_of_nodes() > 130:
    #     return 5

    return 1
