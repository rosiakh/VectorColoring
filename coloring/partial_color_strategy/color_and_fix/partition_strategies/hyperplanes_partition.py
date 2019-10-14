import logging
import math

from scipy.stats import ortho_group

from coloring.partial_color_strategy.color_and_fix.color_and_fix_helper import *
from solver.solver import compute_vector_coloring


def hyperplanes_partition_strategy(graph, partition_strategy_params, shmem_partitions=None, lock=None):
    """Returns a partition (possibly illegal coloring of all vertices) obtained using hyperplanes partition strategy.

    The function returns best of possibly many partitions tried. Obtaining a single partition is delegated to another
    function.

    :param graph: (networkx.Graph) Graph to be colored
    :param partition_strategy_params: parameters needed for chosen partition strategy
    :param shmem_partitions: used for parallel computing only
    :param lock: used for parallel computing only
    :return best partition: (dict) Assignment of colors to every vertex of graph given by partition of vector space.
        Coloring might be illegal.
    """

    vector_coloring = compute_vector_coloring(graph, partition_strategy_params['sdp_type'])
    best_partition = None

    for _ in range(partition_strategy_params['nr_of_partitions_per_vc']):
        temp_colors = partition_by_hyperplanes(graph, partition_strategy_params, vector_coloring)

        if is_better_partition(graph, temp_colors, best_partition,
                               partition_strategy_params['independent_set_extraction_strategy']):
            best_partition = temp_colors

    if shmem_partitions is not None and lock is not None:
        lock.acquire()
        shmem_partitions.append(best_partition)
        lock.release()

    return best_partition


def partition_by_hyperplanes(graph, partition_strategy_params, vector_coloring):
    """Return one partition (possibly illegal coloring of all vertices) of graph.

    :param graph: (nx.Graph)
    :param partition_strategy_params: parameters needed for partition strategy
    :param vector_coloring: (2-dim array) Rows of vector_coloring constitute vector coloring of graph.
    :return: possibly illegal coloring of all vertices obtained using hyperplanes partition strategy
    """

    nr_of_hyperplanes = optimal_nr_of_hyperplanes(graph, vector_coloring)

    n = graph.number_of_nodes()
    hyperplanes_sides = {v: 0 for v in range(0, n)}
    r_arr = get_random_vectors(n, strategy=partition_strategy_params['normal_vectors_generation_strategy'])
    for i in range(nr_of_hyperplanes):
        r = r_arr[i]
        x = np.sign(np.dot(vector_coloring, r))
        for v in range(0, n):
            if x[v] >= 0:
                hyperplanes_sides[v] = hyperplanes_sides[v] * 2 + 1
            else:
                hyperplanes_sides[v] = hyperplanes_sides[v] * 2

    temp_colors = {v: -1 for v in graph.nodes()}
    # Assume that nodes are given in the same order as in rows of vector_coloring
    for i, v in enumerate(sorted(graph.nodes())):
        temp_colors[v] = hyperplanes_sides[i]

    return temp_colors


def optimal_nr_of_hyperplanes(graph, vector_coloring):
    """Returns the optimal number of hyperplanes to use to obtain a partition (possibly illegal coloring of all vertices)
        of graph based on it's vector coloring and (modified) formula from KMS article.

    :param graph: (nx.Graph)
    :param vector_coloring: vector coloring of graph
    :return (int) optimal number of hyperplanes to use for obtaining partition of graph's vertices based on its
        vector coloring
    """

    max_degree = max(dict(graph.degree()).values())
    k = find_number_of_vector_colors_from_vector_coloring(graph, vector_coloring)
    opt_nr_of_hyperplanes = 2
    try:
        opt_nr_of_hyperplanes = 2 + int(math.ceil(math.log(max_degree, k)))
    except ValueError:
        logging.info("math domain error")

    return max(1, opt_nr_of_hyperplanes - 2)


def get_random_vectors(nr_of_vectors, strategy):
    """Returns matrix which rows are random vectors generated according to strategy.

    :param nr_of_vectors: number of random vectors to generate
    :param strategy: strategy of generating random vectors
    :return matrix which rows are random vectors
    """

    if strategy == 'orthonormal':
        array = ortho_group.rvs(nr_of_vectors)
    elif strategy == 'random_normal':
        array = np.zeros((nr_of_vectors, nr_of_vectors))
        for i in range(nr_of_vectors):
            array[i] = np.random.normal(0, 1, nr_of_vectors)
    else:
        raise Exception('Wrong random vector generation strategy')

    return array
