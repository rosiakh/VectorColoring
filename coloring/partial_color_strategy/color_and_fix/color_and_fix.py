import logging

import networkx as nx

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
    """General strategy for coloring whole graph at once and then improving the coloring.

    This strategy colors graph by finding the best partition i.e. possibly illegal coloring of all vertices,
    and then truncating the input graph by finding some proper partial coloring and deleting its vertices.

    Args:
        graph (nx.Graph): Graph to be processed. Function modifies this parameter.
        vector_coloring (2-dim matrix): Rows constitute vector coloring of graph.
        partial_coloring (dict): Dictionary of current (probably partial) partial_coloring of working graph. Function modifies this parameter.
        partition_strategy (lambda graph, vector_coloring, partial_coloring, options): Function that computes coloring, possibly illegal,
            of graph using some hyperplane partition strategy
    """

    logging.info('Looking for partial coloring using color and fix strategy...')

    best_partition = None
    nr_of_trials = 1 if partial_color_strategy_params['deterministic'] else \
        partial_color_strategy_params['nr_of_vector_colorings_to_try']

    for _ in range(nr_of_trials):
        partition = partition_strategy_map[partial_color_strategy_params['partition_strategy']](
            graph, partial_color_strategy_params['partition_strategy_params'])

        if is_better_partition(graph, partition, best_partition,
                               partial_color_strategy_params['independent_set_extraction_strategy']):
            best_partition = partition

    nr_of_nodes_colored = update_coloring_and_graph(
        graph, partial_coloring, best_partition, partial_color_strategy_params['independent_set_extraction_strategy'])

    logging.info("Colored {0} nodes using {1} colors".format(nr_of_nodes_colored, len(set(best_partition.values()))))


def update_coloring_and_graph(graph, partial_coloring, partition, strategy):
    """Given best partition updates global partial_coloring (so that coloring is never illegal) and truncates graph graph

    Args:
        graph (nx.Graph): Graph that is being colored. The function removes some nodes from it so that only the part
            that is not yet colored remains.
        partial_coloring (dict): Global dictionary of colors of vertices of graph.
        partition (dict): Coloring of vertices of graph given by hyperplane partition. Might be illegal.

        :return number of actually colored nodes
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
