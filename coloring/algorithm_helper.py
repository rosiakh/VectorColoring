import random
from operator import itemgetter

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.algorithms import approximation
from scipy.cluster.hierarchy import dendrogram


def find_number_of_vector_colors_from_vector_coloring(graph, vector_coloring):
    """Given vector coloring find number of 'vector-colors' used i.e. smallest 'k' such that vector_coloring is
        vector k-coloring of graph.

        Vector coloring is obtained from matrix coloring computed using SDP optimization and Cholesky factorization.
            Both of those processes may introduce error and return number of 'vector-colors' greater than vector chromatic
            number.

    :param graph: (nx.Graph) Graph of which vector_coloring is vector coloring.
    :param vector_coloring: (2-dim matrix) Rows of vector_coloring comprise vector coloring of graph. Nth row is assigned to
        nth vertex which doesn't have to be vertex 'n'.
    :return (int) smallest 'k' such that vector_coloring is vector k-coloring of graph.
    """

    m = np.dot(vector_coloring, np.transpose(vector_coloring))

    vertices_mapping = {v: i for i, v in enumerate(sorted(graph.nodes()))}
    cells = []
    for i, j in graph.edges():
        cells.append(m[vertices_mapping[i], vertices_mapping[j]])

    mc = max(cells)
    mc = mc if mc < 0 else -mc
    k = 1 - 1 / mc

    return k


def has_edge_between_ith_and_jth(graph, i, j):
    """Checks if there is an edge in graph between i-th vertex and j-th vertex after sorting them.

        Graph may have vertex called 'i' that isn't it's i-th vertex in sorted order (e.g. when some vertices have been
            removed from the graph). This function checks if there is an edge between i-th and j-th vertex in sorted
            order, so i-th and j-th vertex exist as long as those numbers are less than G.number_of_nodes()

    :param graph: (nx.Graph) Graph to be processed
    :param i: (int) Number between 0 and G.number_of_nodes()-1
    :param j: (int) Number between 0 and G.number_of_nodes()-1
    :return (bool) True iff there is an edge in G between i-th vertex and j-th vertex after sorting them.
    """

    return graph.has_edge(sorted(list(graph.nodes()))[i], sorted(list(graph.nodes()))[j])


def check_if_coloring_legal(graph, partial_coloring, partial=False):
    """Checks if given coloring is a legal vertex coloring.

    :param graph: (graph) Colored graph.
    :param partial_coloring: (dict) Global vertex-color dictionary.
    :param partial: (bool) If True, than we check for legal partial coloring, i.e. vertices with both endpoints=-1
                are legal
    :return (bool) True iff coloring of G given by partition is legal (partial) vertex coloring.
    """

    for i, j in graph.edges():
        if partial_coloring[i] == partial_coloring[j]:
            if partial_coloring[i] == -1 and partial:
                continue
            else:
                return False

    return True


def get_nodes_sorted_by_degree(graph):
    """Returns list of nodes sorted by degree in descending order.

    :param graph: (nx.Graph)
    :return list of nodes sorted by degree in decreasing order
    #TODO: it would be enough to just find the highest degree - that's how it's used
    """

    return [item[0] for item in sorted(list(graph.degree(graph.nodes())), key=lambda item: item[1], reverse=True)]


def get_highest_degree_node(graph):
    """Get node with the highest degree.

    :param graph: (nx.Graph) a graph
    :return: node of the graph with the highest degree
    # TODO: use this function instead of sorting all nodes by 'get_nodes_sorted_by_degree'
    """

    return max(list(graph.degree(graph.nodes())), key=itemgetter(1))[0]


def extract_independent_subset(vertices, edges, strategy='max_degree_first'):
    """Returns subset of vertices that constitute an independent set by finding vertices that need to be removed
        to leave an independent set.

    :param vertices: set of vertices
    :param edges: set of edges between vertices
    :param strategy: strategy of extracting independent set
    :return ind_set: set of vertices that constitute an independent set
    """

    subgraph = nx.Graph()
    subgraph.add_nodes_from(vertices)
    subgraph.add_edges_from(edges)
    temp_colors = {v: 0 for v in subgraph.nodes()}

    nodes_to_del = find_nodes_to_delete(subgraph, temp_colors, strategy)

    subgraph.remove_nodes_from(nodes_to_del)
    ind_set = set(subgraph.nodes())

    return ind_set


def find_nodes_to_delete(graph, partition, independent_set_extraction_strategy):
    """Given graph and its possibly illegal coloring returns nodes that should be deleted to obtain legal coloring.

    :param graph: (nx.Graph)
    :param partition: Full, possibly illegal, coloring of graph. If it is None than assume all colors are the same
            (fixed graph should become an independent set).
    :param independent_set_extraction_strategy
    :return nodes to delete so that subgraph induced on remaining nodes will be legally colored
    """

    nodes_to_delete = []

    if partition is None:
        return nodes_to_delete

    # find subgraph induced on all edges that have endpoints with the same color
    illegal_edges = {(i, j) for (i, j) in graph.edges() if
                     partition[i] == partition[j] and partition[i] != -1}
    subgraph_illegal_edges = nx.Graph()
    subgraph_illegal_edges.add_edges_from(illegal_edges)

    # find set of vertices that, when deleted from graph, leaves it properly colored
    if independent_set_extraction_strategy == 'min_vertex_cover':
        nodes_to_delete = get_nodes_to_delete_min_vertex_cover(subgraph_illegal_edges)
    elif independent_set_extraction_strategy == 'arora_kms':
        nodes_to_delete = get_nodes_to_delete_arora_kms(subgraph_illegal_edges)
    elif independent_set_extraction_strategy == 'arora_kms_prim':
        nodes_to_delete = get_nodes_to_delete_arora_kms_prim(subgraph_illegal_edges)
    elif independent_set_extraction_strategy == 'max_degree_first':
        nodes_to_delete = get_nodes_to_delete_max_degree_first(subgraph_illegal_edges)
    else:
        raise Exception('Unknown node fixing strategy')

    return nodes_to_delete


def get_nodes_to_delete_min_vertex_cover(subgraph_illegal_edges):
    """Find vertex cover using approximate minimum vertex cover algorithm.

    :param subgraph_illegal_edges: (nx.Graph) graph will all edges monochromatic (each edge may have different color)
    :return: subset of nodes of subgraph_illegal_edges that contains at least one vertex from each edge (vertex cover)
    """

    return approximation.min_weighted_vertex_cover(subgraph_illegal_edges)


def get_nodes_to_delete_arora_kms(subgraph_illegal_edges):
    """Return set of all vertices as vertex cover.

    :param subgraph_illegal_edges: (nx.Graph) graph will all edges monochromatic (each edge may have different color)
    :return: subset of nodes of subgraph_illegal_edges that contains at least one vertex from each edge (vertex cover)
    """

    return subgraph_illegal_edges.nodes()


def get_nodes_to_delete_arora_kms_prim(subgraph_illegal_edges):
    """Find vertex cover by iteratively removing random edge from subgraph_illegal_edges until no edge remains

    :param subgraph_illegal_edges: (nx.Graph) graph will all edges monochromatic (each edge may have different color)
    :return: subset of nodes of subgraph_illegal_edges that contains at least one vertex from each edge (vertex cover)
    """

    nodes_to_delete = []
    nodes_by_degree = get_nodes_sorted_by_degree(subgraph_illegal_edges)
    while nodes_by_degree and subgraph_illegal_edges.degree[nodes_by_degree[0]] > 0:
        edge_to_delete = random.choice(list(subgraph_illegal_edges.edges()))
        subgraph_illegal_edges.remove_nodes_from(edge_to_delete)
        nodes_to_delete.extend(edge_to_delete)
        nodes_by_degree = get_nodes_sorted_by_degree(subgraph_illegal_edges)
    return nodes_to_delete


def get_nodes_to_delete_max_degree_first(subgraph_illegal_edges):
    """Find vertex cover by iteratively removing node with the highest degree from subgraph_illegal_edges
        until no edge remains.

    :param subgraph_illegal_edges: (nx.Graph) graph will all edges monochromatic (each edge may have different color)
    :return: subset of nodes of subgraph_illegal_edges that contains at least one vertex from each edge (vertex cover)
    """

    nodes_to_delete = []
    nodes_by_degree = get_nodes_sorted_by_degree(subgraph_illegal_edges)
    while nodes_by_degree and subgraph_illegal_edges.degree[nodes_by_degree[0]] > 0:
        nodes_to_delete.append(nodes_by_degree[0])
        subgraph_illegal_edges.remove_node(nodes_by_degree[0])
        nodes_by_degree = get_nodes_sorted_by_degree(subgraph_illegal_edges)
    return nodes_to_delete


def get_lowest_legal_color(graph, vertex, coloring):
    """Gets lowest color that can be used to legally color vertex

    :param graph: nx.Graph object
    :param vertex: vertex of graph
    :param coloring: partial coloring of graph
    :return: Lowest color that can be used to legally color vertex
    """

    taken_colors = {coloring[v] for v in graph.neighbors(vertex)}
    for color in range(0, graph.number_of_nodes()):
        if color not in taken_colors:
            return color

    return len(taken_colors)


def show_dendrogram(z):
    # TODO: see if it's used and update docstring

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        z,
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()
