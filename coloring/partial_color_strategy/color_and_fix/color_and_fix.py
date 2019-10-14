import logging

from coloring.partial_color_strategy.color_and_fix.color_and_fix_helper import *
from coloring.partial_color_strategy.color_and_fix.partition_strategies.clustering_partition import \
    clustering_partition_strategy, kmeans_clustering_partition_strategy
from coloring.partial_color_strategy.color_and_fix.partition_strategies.hyperplanes_partition import \
    hyperplanes_partition_strategy

np.set_printoptions(precision=5, suppress=True)

partition_strategy_map = {
    'hyperplane_partition': hyperplanes_partition_strategy,
    'clustering': clustering_partition_strategy,
    'kmeans_clustering': kmeans_clustering_partition_strategy,
    None: None,
}


def color_all_vertices_at_once(graph, partial_coloring, partial_color_strategy_params):
    """ General strategy for coloring whole graph at once and then improving the coloring.

    This strategy colors graph by finding the best partition i.e. possibly illegal coloring of all vertices,
    and then truncating the input graph by finding some proper partial coloring and deleting its vertices.

    :param graph: (nx.Graph) Graph to be processed. Function modifies this parameter.
    :param partial_coloring: (dict) Dictionary of current partial_coloring of graph. Function modifies this parameter.
    :param partial_color_strategy_params: (dict) params of chosen partial color strategy (here: color_and_fix)
    """

    logging.info('Looking for partial coloring using color and fix strategy...')

    best_partition = None
    nr_of_trials = 1 if partial_color_strategy_params['deterministic'] else \
        partial_color_strategy_params['nr_of_vector_colorings_to_try']

    for _ in range(nr_of_trials):
        # find partition i.e. possibly illegal coloring of ALL vertices
        partition = partition_strategy_map[partial_color_strategy_params['partition_strategy']](
            graph, partial_color_strategy_params['partition_strategy_params'])

        if is_better_partition(graph, partition, best_partition,
                               partial_color_strategy_params['independent_set_extraction_strategy']):
            best_partition = partition

    # find some proper partial coloring as a subset of found best partition and update partial_coloring of graph
    # accordingly. remove colored vertices from graph
    nr_of_nodes_colored = update_coloring_and_graph(
        graph, partial_coloring, best_partition, partial_color_strategy_params['independent_set_extraction_strategy'])

    logging.info("Colored {0} nodes using {1} colors".format(nr_of_nodes_colored, len(set(best_partition.values()))))


def update_coloring_and_graph(graph, partial_coloring, partition, strategy):
    """ Given (best found) partition updates global partial_coloring (so that it is never illegal) and truncates graph
        by removing colored vertices.

    :param graph: (nx.Graph) Graph that is being colored. The function removes some nodes from it so that only the part
            that is not yet colored remains.
    :param partial_coloring: (dict) Global dictionary of colors of vertices of graph.
    :param partition: (dict) Coloring of vertices of graph given by hyperplane partition. Might be illegal.
    :param strategy: strategy of removing nodes to obtain legal coloring from a partition
    :return number of actually colored nodes (=number of vertices removed from graph)
    """

    nodes_to_del = find_nodes_to_delete(graph, partition, strategy)
    nodes_to_color = {n for n in graph.nodes() if n not in nodes_to_del}
    min_color = max(partial_coloring.values()) + 1
    for v in nodes_to_color:
        partial_coloring[v] = min_color + partition[v]

    if not check_if_coloring_legal(graph, partial_coloring, partial=True):
        raise Exception('Some partition resulted in illegal coloring.')

    graph.remove_nodes_from(nodes_to_color)

    return len(nodes_to_color)
