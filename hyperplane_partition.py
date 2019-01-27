import logging
import math

import networkx as nx
import networkx.algorithms.approximation
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np

from algorithm_helper import *

np.set_printoptions(precision=5, suppress=True)


def better_partition(g, part1, part2):
    """Checks whether the first partition is better than the second one."""

    # TODO: When there are more hyperplanes it often chooses the resulting partition
    # TODO: as best even though it results in more colors (e.g. for DSJC 125.5)

    if part2 is None or len(part2) == 0:
        return True

    if part1 is None or len(part1) == 0:
        return False

    # Remove colors from one endpoint of each illegal edge in each partition.
    nodes_to_delete1 = nodes_to_delete(g, part1, strategy='max_degree_first')
    nodes_to_color1 = {n for n in g.nodes() if n not in nodes_to_delete1}
    nr_of_colors1 = len(set(part1.values()))

    nodes_to_delete2 = nodes_to_delete(g, part2, strategy='max_degree_first')
    nodes_to_color2 = {n for n in g.nodes() if n not in nodes_to_delete2}
    nr_of_colors2 = len(set(part2.values()))

    avg1 = float(len(nodes_to_color1)) / nr_of_colors1
    avg2 = float(len(nodes_to_color2)) / nr_of_colors2

    return avg1 > avg2


def partially_color_graph_by_hyperplane_partition(g, L, colors, partition_strategy):
    """General strategy for using hyperplane partition of vector space.

    Args:
        partition_strategy (lambda g, L, colors, options): Function that computes coloring, possibly illegal, of g
            using some hyperplane partition strategy
        options_creator (lambda g, L, colors): Creates list of options for a given partition strategy
    """

    def update_colors_and_graph(g, colors, partition):
        """Given best partition updates global colors (so that coloring is never illegal) and truncates graph g

        Args:
            g (nx.Graph): Graph that is being colored. The function removes some nodes from it so that only the part
                that is not yet colored remains.
            colors (dict): Global dictionary of colors of vertices of g.
            partition (dict): Coloring of vertices of g given by hyperplane partition. Might be illegal.
        """

        nodes_to_del = nodes_to_delete(g, partition, strategy='max_degree_first')
        nodes_to_color = {n for n in g.nodes() if n not in nodes_to_del}

        min_color = max(colors.values()) + 1
        for v in nodes_to_color:
            colors[v] = min_color + partition[v]

        if not check_if_coloring_legal(g, colors, partial=True):
            raise Exception('Some hyperplane partition resulted in illegal coloring.')

        g.remove_nodes_from(nodes_to_color)

        logging.info('There are {0} vertices left to color'.format(g.number_of_nodes()))

    logging.info('Looking for partial coloring using some hyperplane partition strategy...')

    # Config
    iterations = 1

    best_partition = None
    for it in range(iterations):
        partition = partition_strategy(g, L)

        if better_partition(g, partition, best_partition):
            best_partition = partition

    update_colors_and_graph(g, colors, best_partition)


def random_partition_strategy(g, L):
    """Returns the result of single partition using random hyperplane strategy.

    Args:
        g (networkx.Graph): Graph to be colored
        L (2-dim array): Rows of L constitute vector coloring of g.

    Returns:
        dict: Assignment of colors to every vertex of g given by partition of vector space. Coloring might be illegal.
    """

    def optimal_nr_of_hyperplanes(g, L):
        """Returns the optimal number of hyperplanes.

        Returns:
            opt_nr_of_hyperplanes (int)
        """

        max_degree = max(dict(g.degree()).values())
        k = find_number_of_vector_colors_from_vector_coloring(g, L)
        opt_nr_of_hyperplanes = 2 + int(math.ceil(math.log(max_degree, k)))

        return max(1, opt_nr_of_hyperplanes - 2)

    # Config
    nr_of_hyperplanes = optimal_nr_of_hyperplanes(g, L)

    n = g.number_of_nodes()
    hyperplanes_sides = {v: 0 for v in range(0, n)}
    for i in range(nr_of_hyperplanes):
        r = np.random.normal(0, 1, n)
        x = np.sign(np.dot(L, r))
        for v in range(0, n):
            if x[v] >= 0:
                hyperplanes_sides[v] = hyperplanes_sides[v] * 2 + 1
            else:
                hyperplanes_sides[v] = hyperplanes_sides[v] * 2

    temp_colors = {v: -1 for v in g.nodes()}
    for i, v in enumerate(sorted(g.nodes())):  # Assume that nodes are given in the same order as in rows of L
        temp_colors[v] = hyperplanes_sides[i]

    return temp_colors


def orthogonal_partition_strategy(g, L):
    """Returns the result of single partition using random hyperplane strategy.

    Args:
        g (networkx.Graph): Graph to be colored
        L (2-dim array): Rows of L constitute vector coloring of g.

    Returns:
        dict: Assignment of colors to every vertex of g given by partition of vector space. Coloring might be illegal.
    """

    def optimal_nr_of_hyperplanes(g, L):
        """Returns the optimal number of hyperplanes.

        Returns:
            opt_nr_of_hyperplanes (int)
        """

        max_degree = max(dict(g.degree()).values())
        k = find_number_of_vector_colors_from_vector_coloring(g, L)
        opt_nr_of_hyperplanes = 2 + int(math.ceil(math.log(max_degree, k)))

        return min(opt_nr_of_hyperplanes, g.number_of_nodes())

    # Config
    nr_of_hyperplanes = optimal_nr_of_hyperplanes(g, L)

    n = g.number_of_nodes()
    hyperplanes_sides = {v: 0 for v in range(0, n)}
    for i in range(nr_of_hyperplanes):
        r = np.zeros(n)
        r[i] = 1.0
        x = np.sign(np.dot(L, r))
        for v in range(0, n):
            if x[v] >= 0:
                hyperplanes_sides[v] = hyperplanes_sides[v] * 2 + 1
            else:
                hyperplanes_sides[v] = hyperplanes_sides[v] * 2

    temp_colors = {v: -1 for v in g.nodes()}
    for i, v in enumerate(sorted(g.nodes())):  # Assume that nodes are given in the same order as in rows of L
        temp_colors[v] = hyperplanes_sides[i]

    return temp_colors


def clustering_partition_strategy(g, L):
    Z = linkage(L, method='complete', metric='cosine')

    # show_dendrogram(Z)

    k = find_number_of_vector_colors_from_vector_coloring(g, L)
    opt_t = 1 + 1 / (
            k - 1) - 0.001 if k > 1.5 else 2.0  # Should guarantee that each cluster can be colored with one color
    # t *= 1.1  # Make clusters a bit bigger

    best_partition = None
    for t in np.linspace(opt_t - 0.3, opt_t + 0.3, 20):
        clusters = fcluster(Z, t, criterion='distance')
        partition = {n: clusters[v] for v, n in enumerate(sorted(list(g.nodes())))}
        if better_partition(g, partition, best_partition):
            best_partition = partition

    return best_partition


def kmeans_clustering_partition_strategy(g, L):
    from spherecluster import SphericalKMeans

    n = g.number_of_nodes()
    best_partition = None
    for k in range(int(0.2 * n) + 1, int(0.5 * n), 5):
        # clusters = KMeans(n_clusters=k).fit_predict(L)
        # partition = {n: clusters[v] for v, n in enumerate(sorted(list(g.nodes())))}
        skm = SphericalKMeans(n_clusters=k).fit(L)
        partition = {n: skm.labels_[v] for v, n in enumerate(sorted(list(g.nodes())))}
        if better_partition(g, partition, best_partition):
            best_partition = partition

    return best_partition
