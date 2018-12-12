import os

import numpy as np
from matplotlib import pyplot as plt
from networkx import Graph
from scipy.cluster.hierarchy import dendrogram
import networkx as nx

dirname = '/home/hubert/VectorColoring/'
if not os.path.exists(dirname):
    os.makedirs(dirname)

vc_dirname = dirname + 'VectorColorings/'
if not os.path.exists(vc_dirname):
    os.makedirs(vc_dirname)


def find_number_of_vector_colors_from_vector_coloring(g, L):
    """Given vector coloring find number of 'vector-colors' used i.e. smallest 'k' such that L is vector k-coloring of g.

        Vector coloring is obtained from matrix coloring computed using SDP optimalization and Cholesky factorization.
            Both of those processes may introduce error and return number of 'vector-colors' greater than vector chromatic
            number.

        Args:
            g (Graph): Graph of which L is vector coloring.
            L (2-dim matrix): Rows of L are vector coloring of g. Nth row is assigned to nth vertex which doesn't
                have to be vertex 'n'.

        Returns:
            int: smallest 'k' such that L is vector k-coloring of g.
        """

    M = np.dot(L, np.transpose(L))

    vertices_mapping = {v: i for i, v in enumerate(sorted(g.nodes()))}
    cells = []
    for i, j in g.edges():
        cells.append(M[vertices_mapping[i], vertices_mapping[j]])

    return 1 - 1 / max(cells)


def check_if_coloring_legal(g, colors, partial=False):
    """Checks if given coloring is a legal vertex coloring.

        Args:
            g (graph): Colored graph.
            colors (dict): Global vertex-color dictionary.
            partial (bool): If True, than we check for legal partial colloring, i.e. vertices with both endpoints=-1
                are legal

        Returns:
            bool: True iff coloring of G given by colors is legal (partial) vertex coloring.
        """

    for i, j in g.edges():
        if colors[i] == colors[j]:
            if colors[i] == -1 and partial:
                continue
            else:
                return False

    return True


def get_nodes_sorted_by_degree(g):
    """Returns list of nodes sorted by degree in descending order."""

    return [item[0] for item in
            sorted(list(g.degree(g.nodes())), key=lambda item: item[1], reverse=True)]


def nodes_to_delete(g, colors, strategy='max_degree_first'):
    """Given graph and its possibly illegal coloring returns nodes that should be deleted to obtain legal coloring.

    Args:
        colors (dict): Full coloring of g. If it is None than assume all colors are the same
            (fixed g should become an independent set).
    """

    nodes_to_delete = []

    illegal_edges = {(i, j) for (i, j) in g.edges() if colors[i] == colors[j] and colors[i] != -1}
    subgraph_illegal_edges = nx.Graph()
    subgraph_illegal_edges.add_edges_from(illegal_edges)

    if strategy == 'min_vertex_cover':
        nodes_to_delete = nx.algorithms.approximation.min_weighted_vertex_cover(subgraph_illegal_edges)
    elif strategy == 'max_degree_first':
        nodes_by_degree = get_nodes_sorted_by_degree(subgraph_illegal_edges)

        while nodes_by_degree:
            if subgraph_illegal_edges.degree(nodes_by_degree[0]) == 0:
                break
            nodes_to_delete.append(nodes_by_degree[0])
            subgraph_illegal_edges.remove_node(nodes_by_degree[0])
            nodes_by_degree = get_nodes_sorted_by_degree(subgraph_illegal_edges)
    else:
        raise Exception('Unknown node fixing strategy')

    return nodes_to_delete


def get_lowest_legal_color(graph, vertex, coloring):
    """Gets lowest color that can be used to legally color vertex assuming that all its neighbors are already colored
        in coloring."""

    taken_colors = {coloring[v] for v in graph.neighbors(vertex)}
    for clr in range(0, graph.number_of_nodes()):
        if clr not in taken_colors:
            return clr

    return len(taken_colors)


def extract_independent_subset(vertices, edges):
    """Returns subset of vertices that constitute an independent set."""

    subgraph = nx.Graph()
    subgraph.add_nodes_from(vertices)
    subgraph.add_edges_from(edges)
    temp_colors = {v: 0 for v in subgraph.nodes()}

    nodes_to_del = nodes_to_delete(subgraph, temp_colors, strategy='max_degree_first')

    subgraph.remove_nodes_from(nodes_to_del)

    return set(subgraph.nodes())


def show_dendrogram(Z):
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()


def get_vector_coloring_filename(g, strict):
    vc_dirname = dirname + 'VectorColorings/'
    filename = vc_dirname + '_' + g.name + '_strict=' + str(strict) + '_VectorColoring'

    return filename


def save_vector_coloring_to_file(g, strict, l):
    filename = get_vector_coloring_filename(g, strict)
    np.savetxt(filename, l)


def vector_coloring_in_file(g, strict):
    filename = get_vector_coloring_filename(g, strict)

    return os.path.exists(filename)


def read_vector_coloring_from_file(g, strict):
    filename = get_vector_coloring_filename(g, strict)

    return np.loadtxt(filename)
