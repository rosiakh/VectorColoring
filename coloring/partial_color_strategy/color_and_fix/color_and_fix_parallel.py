from multiprocessing import Lock, Process, Manager

import networkx as nx

from color_and_fix import *
from configuration import algorithm_options_config
from solver.solver import compute_vector_coloring


def color_all_vertices_at_once_parallel(graph, partial_coloring, partial_color_strategy_params):
    """General strategy for coloring whole graph at once and then improving the coloring.

    This strategy colors graph by finding the best partition i.e. possibly illegal coloring of all vertices,
    and then truncating the input graph by finding some proper partial coloring and deleting its vertices.

    Args:
        graph (nx.Graph): Graph to be processed. Function modifies this parameter.
        vector_coloring (2-dim matrix): Rows constitute vector coloring of graph.
        partial_coloring (dict): Dictionary of current (probably partial) colors of working graph. Function modifies this parameter.
        partition_strategy (lambda graph, vector_coloring, colors, options): Function that computes coloring, possibly illegal,
            of graph using some hyperplane partition strategy
    """

    logging.info('Looking for partial coloring using all_vertices_at_once strategy...')

    vector_coloring = compute_vector_coloring(graph, partial_color_strategy_params['sdp_type'])

    best_partition = None

    manager = Manager()
    shmem_partitions = manager.list()
    lock = Lock()
    processes = []

    iterations = 1 if partial_color_strategy_params['deterministic'] else \
        partial_color_strategy_params['nr_of_partitions_to_try'] / algorithm_options_config.nr_of_parallel_jobs
    nr_jobs = 1 if partial_color_strategy_params['deterministic'] else algorithm_options_config.nr_of_parallel_jobs

    for _ in range(iterations):
        for _ in range(nr_jobs):
            processes.append(
                Process(target=partition_strategy_map[partial_color_strategy_params['partition_strategy']],
                        args=(graph, vector_coloring, partial_color_strategy_params, shmem_partitions, lock)))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        processes = []

    best_partition = better_partition_parallel(graph, shmem_partitions, best_partition,
                                               partial_color_strategy_params['independent_set_extraction_strategy'])

    update_coloring_and_graph(graph, partial_coloring, best_partition,
                              partial_color_strategy_params['independent_set_extraction_strategy'])

    logging.info('Partial coloring found. There are {0} vertices left to color'.format(graph.number_of_nodes()))


def update_coloring_and_graph(graph, colors, partition, strategy):
    """Given best partition updates global partial_coloring (so that coloring is never illegal) and truncates graph graph

    Args:
        graph (nx.Graph): Graph that is being colored. The function removes some nodes from it so that only the part
            that is not yet colored remains.
        colors (dict): Global dictionary of colors of vertices of graph.
        partition (dict): Coloring of vertices of graph given by hyperplane partition. Might be illegal.
    """

    nodes_to_del = find_nodes_to_delete(graph, partition, strategy=strategy)
    nodes_to_color = {n for n in graph.nodes() if n not in nodes_to_del}

    min_color = max(colors.values()) + 1
    for v in nodes_to_color:
        colors[v] = min_color + partition[v]

    if not check_if_coloring_legal(graph, colors, partial=True):
        raise Exception('Some partition resulted in illegal coloring.')

    graph.remove_nodes_from(nodes_to_color)


def better_partition_parallel(graph, part1, part2, independent_set_extraction_strategy):
    """ part2 i current best & part1 is shmeme list"""

    best = part2
    for i in range(len(part1)):
        if is_better_partition(graph, part1[i], best, independent_set_extraction_strategy):
            best = part1[i]

    return best
