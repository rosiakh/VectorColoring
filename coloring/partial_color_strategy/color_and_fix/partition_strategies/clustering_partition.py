from scipy.cluster.hierarchy import linkage, fcluster

from coloring.partial_color_strategy.color_and_fix.color_and_fix import *
from solver.solver import compute_vector_coloring


def clustering_partition_strategy(graph, partition_strategy_params, shmem_partitions=None, lock=None):
    """Finds a good partition - possibly illegal coloring of all vertices of graph - using clustering strategy.

    Clustering strategy is based on hierarchical clustering and dividing vertices into cluster of which every one
    is colored using different color. Each vertex initially is it's own cluster but in every iteration clusters
    closest to each other are joined to create a new bigger cluster. This joining operation stops when the first cluster
    exceeds given size limit.

    :param graph: (nx.Graph) graph
    :param partition_strategy_params: parameters for partition strategy
    :param shmem_partitions: used for parallel computing only
    :param lock: used for parallel computing only
    :return: best found partition - coloring, possibly illegal, of all graph's vertices
    """

    vector_coloring = compute_vector_coloring(graph, partition_strategy_params['sdp_type'])

    z = linkage(vector_coloring, method='complete', metric='cosine')

    # computing opt_t this way should guarantee that each cluster can be colored with one color
    k = find_number_of_vector_colors_from_vector_coloring(graph, vector_coloring)
    opt_t = 1 + 1 / (k - 1) - 0.001 if k > 1.5 else 2.0

    best_partition = None
    for t in np.linspace(
            opt_t * partition_strategy_params['cluster_size_lower_factor'],
            opt_t * partition_strategy_params['cluster_size_upper_factor'],
            partition_strategy_params['nr_of_cluster_sizes_to_check']):
        clusters = fcluster(z, t, criterion='distance')
        partition = {n: clusters[v] for v, n in enumerate(sorted(list(graph.nodes())))}
        if is_better_partition(graph, partition, best_partition,
                               partition_strategy_params['independent_set_extraction_strategy']):
            best_partition = partition

    if shmem_partitions is not None and lock is not None:
        lock.acquire()
        shmem_partitions.append(best_partition)
        lock.release()

    return best_partition


def kmeans_clustering_partition_strategy(graph, partition_strategy_params, shmem_partitions=None, lock=None):
    """
    :param graph: (nx.Graph) a graph
    :param partition_strategy_params: all the parameters needed for partition strategy
    :param shmem_partitions: used for parallel computing only
    :param lock: used for parallel computing only
    :return: best found partition - coloring, possibly illegal, of all graph's vertices
    """
    # TODO: probably need rework and update to docstrings

    from spherecluster import SphericalKMeans

    vector_coloring = compute_vector_coloring(graph, partition_strategy_params['sdp_type'])

    n = graph.number_of_nodes()
    best_partition = None
    for k in range(int(0.2 * n) + 1, int(0.5 * n), 5):
        skm = SphericalKMeans(n_clusters=k).fit(vector_coloring)
        partition = {n: skm.labels_[v] for v, n in enumerate(sorted(list(graph.nodes())))}
        if is_better_partition(graph, partition, best_partition,
                               partition_strategy_params['independent_set_extraction_strategy']):
            best_partition = partition

    if shmem_partitions is not None and lock is not None:
        lock.acquire()
        shmem_partitions.append(best_partition)
        lock.release()

    return best_partition
