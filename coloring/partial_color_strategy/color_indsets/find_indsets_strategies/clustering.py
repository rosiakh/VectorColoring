from scipy.cluster.hierarchy import linkage, fcluster

from coloring.algorithm_helper import *
from coloring.partial_color_strategy.color_indsets.color_indsets_helper import *
from solver.solver import compute_vector_coloring


def find_ind_sets_by_clustering(graph, find_indsets_strategy_params, nr_of_sets=1, shmem_ind_sets=None, lock=None):
    """Returns independent sets. Tries to return nr_of_sets but might return less.

    :param graph: (nx.Graph)
    :param find_indsets_strategy_params: parameters of chosen strategy of finding independent sets
    :param nr_of_sets: number of independent sets of vertices of graph that the function tries to find but might
        return less
    :param shmem_ind_sets: used for parallel computation only
    :param lock: used for parallel computation only
    :return (best_ind_sets, best_almost_ind_sets):
        best_ind_sets: list of independent sets of vertices, each of which was extracted from one cluster
        best_almost_ind_sets: list of sets of vertices, each of which is one cluster that was used to obtain the
            corresponding vertex set from best_ind_sets
    """

    vector_coloring = compute_vector_coloring(graph, find_indsets_strategy_params['sdp_type'])

    z = linkage(vector_coloring, method='complete', metric='cosine')

    k = find_number_of_vector_colors_from_vector_coloring(graph, vector_coloring)
    opt_t = 1 + 1 / (k - 1) - 0.001 if k > 1.5 else 2.0  # Should guarantee each cluster can be colored with one color

    best_ind_sets = None
    best_almost_ind_sets = None
    # try different maximal cluster sizes
    for t in np.linspace(
            opt_t * find_indsets_strategy_params['cluster_size_lower_factor'],
            opt_t * find_indsets_strategy_params['cluster_size_upper_factor'],
            num=find_indsets_strategy_params['nr_of_cluster_sizes_to_check']):

        # obtain clusters using hierarchical clustering and given maximal cluster size
        clusters = fcluster(z, t, criterion='distance')
        # for each vertex number associate it with it's cluster
        vertex_nr_to_cluster_nr = {n: clusters[v] for v, n in enumerate(sorted(list(graph.nodes())))}

        cluster_sizes = {}
        for key, value in vertex_nr_to_cluster_nr.items():
            if value not in cluster_sizes:
                cluster_sizes[value] = 1
            else:
                cluster_sizes[value] += 1

        # sort (cluster_nr, cluster_size) pairs in descending order of sizes
        sorted_cluster_sizes = sorted(cluster_sizes.items(), key=lambda (clst_nr, clst_size): clst_size, reverse=True)

        ind_sets = []
        almost_ind_sets = []
        # for descending cluster sizes - get vertices belonging to a cluster as a 'almost_ind_set' and then extract
        # independent set from it
        for i in range(min(nr_of_sets, len(sorted_cluster_sizes))):
            cluster_nr = sorted_cluster_sizes[i][0]

            # get vertices belonging to the cluster
            vertices = {v for v, clr in vertex_nr_to_cluster_nr.items() if clr == cluster_nr}
            edges = nx.subgraph(graph, vertices).edges()
            ind_set = extract_independent_subset(
                vertices, edges, strategy=find_indsets_strategy_params['independent_set_extraction_strategy'])
            ind_sets.append(ind_set)
            almost_ind_sets.append(vertices)

        if is_better_ind_sets(graph, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets
            best_almost_ind_sets = almost_ind_sets

    if shmem_ind_sets is not None and lock is not None:
        lock.acquire()
        shmem_ind_sets.append(best_ind_sets)
        lock.release()

    return best_ind_sets, best_almost_ind_sets
