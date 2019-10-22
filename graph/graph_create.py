"""Module for creating graphs of different families."""

import itertools
import random

import networkx as nx

from configuration.paths_config import generated_graphs_directory
from graph.graph_io import save_graph_to_col_file


def create_kneser_graph(m):
    """Creates Kneser graph that is graph whose vertices correspond to r-element subsets of m-element set and edges
        connect those subsets that have less than t elements in common.

    In this function r and t parameters are set based on m.

    :param m: parameter m from the Kneser graph description
    :return: (nx.Graph) created Kneser graph
    """

    r = m / 2
    t = m / 8

    vertices = [combination for combination in itertools.combinations(range(m), r)]
    edges = []
    for i in range(len(vertices)):
        for j in range(len(vertices)):
            if i < j and len(set(vertices[i]) & set(vertices[j])) < t:
                edges.append((vertices[i], vertices[j]))

    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)

    graph.name = 'Kneser graph {0}'.format(m)

    return graph


def create_k_colorable_graph(k, n, p, cluster_by_modulo=False):
    """Creates k-colorable graph with all color classes of the same cardinality.

    If p < 1 then chromatic number may be less than k.

    :param k: (int) Number of vertex clusters.
    :param n: (int) Number of vertices in each cluster.
    :param p: (float) Probability of edge creation between vertices in different clusters.
    :param cluster_by_modulo: (bool) if true, each cluster contains vertices with the same remainder modulo k
    :return: (nx.Graph) created graph
    """

    vertices = list(range(k * n))
    vertex_cluster = {}

    for v in vertices:
        vertex_cluster[v] = random.choice([c for c in range(k)])

    if cluster_by_modulo:
        for v in vertices:
            vertex_cluster[v] = v % k

    edges = []
    for i in vertices:
        for j in vertices:
            if i > j and vertex_cluster[i] != vertex_cluster[j] and random.random() < p:
                edges.append((i, j))

    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)

    graph.name = 'graph with {0}v in {1} clusters and {2} edge prob'.format(n, k, p)
    graph.family = 'k-colorable'
    graph.starting_index = 0

    return graph


def create_k_cycle(k, n):
    """Creates graph that is hard to color for dsatur.

    :param k: (int) Distance to the farthest neighbor of each vertex.
    :param n: (int) Number of vertices.
    :return: (nx.Graph) created graph
    """

    vertices = [i for i in range(n)]
    edges = []
    for i in vertices:
        for j in vertices:
            if 0 < ((i - j) % n) <= k:
                edges.append((i, j))

    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)

    graph.name = '{0}-cycle {1}v'.format(k, n)
    graph.family = 'k-cycle'
    graph.starting_index = 0

    return graph


def create_crown_graph(n):
    """Creates crown graph with 2n vertices (https://en.wikipedia.org/wiki/Crown_graph)

    :param n: half of the number of vertices
    :return: (nx.Graph) created graph
    """

    vertices = [i for i in range(2 * n)]
    edges = []
    for i in range(n):
        for j in range(n, 2 * n):
            if abs(i - j) != n:
                edges.append((i, j))

    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)

    graph.name = 'crown graph {0}n'.format(n)
    return graph


def create_star(n):
    """Creates star with 1 central node and n nodes of degree 1. (https://en.wikipedia.org/wiki/Star_(graph_theory))

    :param n: one less than number of nodes
    :return: (nx.Graph) created graph
    """

    graph = nx.star_graph(n)
    graph.name = 'star_{0}n'.format(n)
    graph.family = 'star'
    graph.starting_index = 0

    return graph


def create_erdos_renyi_graph(n, p, name_suffix=""):
    """Creates random graph of type networkx.erdos_renyi_graph. (https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model)

    :param n:
    :param p:
    :param name_suffix: suffix added to the name of the graph
    :return (nx.Graph) created graph
    """

    graph = nx.erdos_renyi_graph(n, p)
    graph.name = 'erdos_renyi_{0}n_{1}p'.format(n, p) + '_' + name_suffix
    graph.family = 'erdos_renyi'
    graph.starting_index = 0

    return graph


def create_watts_strogatz_graph(n, k, p, name_suffix=""):
    """Creates random graph of type nx.connected_watts_strogatz_graph.

    :param n: (int) the number of nodes
    :param k: (int) each node is joined with k nearest neighbors in a ring topology
    :param p: (float) the probability of rewiring each edge
    :param name_suffix: suffix added to the name of the graph
    :return (nx.Graph) created graph
    """

    graph = nx.connected_watts_strogatz_graph(n, k, p)
    graph.name = 'watts_strogatz_{0}n_{1}k_{2}p'.format(n, k, p) + '_' + name_suffix
    graph.family = 'watts_strogatz'
    graph.starting_index = 0

    return graph


def create_barabasi_albert_graph(n, m, name_suffix=""):
    """Creates random graph of type networkx.dense_gnm_random_graph.

    :param n: (int) number of nodes
    :param m: (int) number of edges to attach from a new node to existing nodes
    :param name_suffix: suffix added to the name of the graph
    :return (nx.Graph) created graph
    """

    graph = nx.dense_gnm_random_graph(n, m)
    graph.name = 'barabasi_albert_{0}n_{1}m'.format(n, m) + '_' + name_suffix
    graph.family = 'barabasi_albert'
    graph.starting_index = 0

    return graph


def create_set_of_random_graphs(min_vertices=20, max_vertices=40, iterations_per_vertex_number=2):
    """Creates set of random graphs using erdos_renyi model creating iterations_per_vertex_number graphs
        for each value between min_vertices and max_vertices.

    :param min_vertices: minimal number of vertices of each graph
    :param max_vertices: maximal number of vertices of each graph
    :param iterations_per_vertex_number: number of graphs per each number of vertices
    :return: list of graph (nx.Graph)
    """

    graphs = []
    for n in range(min_vertices, max_vertices):
        max_m = (n * (n - 1)) / 2
        density = [0.5 * max_m]
        for m in density:
            for _ in range(iterations_per_vertex_number):
                graphs.append(create_erdos_renyi_graph(n, m))

    return graphs


if __name__ == "__main__":
    nr_of_random_graphs_per_type = 1
    graphs = []

    # Non-random graphs
    k_cycle_params = [(50, 5), (50, 12), (50, 20), (125, 12), (125, 30), (125, 50), (250, 24), (250, 100)]
    for n, k in k_cycle_params:
        graphs.append(create_k_cycle(n=n, k=k))

    # Random graphs
    erdos_renyi_params = [(50, 0.1), (50, 0.5), (50, 0.9), (125, 0.1), (125, 0.5), (125, 0.9), (250, 0.1), (250, 0.5)]
    watts_strogatz_params = [(50, 5, 0.1), (50, 25, 0.1), (50, 40, 0.1), (125, 15, 0.1), (125, 60, 0.1),
                             (125, 100, 0.1), (250, 30, 0.1), (250, 120, 0.1)]
    barabasi_albert_params = [(50, int(0.1 * ((50 * 49) / 2))), (50, int(0.5 * ((50 * 49) / 2))),
                              (50, int(0.9 * ((50 * 49) / 2))), (125, int(0.1 * ((125 * 124) / 2))),
                              (125, int(0.5 * ((125 * 124) / 2))), (125, int(0.9 * ((125 * 124) / 2))),
                              (250, int(0.1 * ((250 * 249) / 2))), (250, int(0.5 * ((250 * 249) / 2))),
                              (250, int(0.9 * ((250 * 249) / 2)))]

    for i in range(nr_of_random_graphs_per_type):
        for n, p in erdos_renyi_params:
            graphs.append(create_erdos_renyi_graph(n=n, p=p, name_suffix=str(i)))

        for n, k, p in watts_strogatz_params:
            graphs.append(create_watts_strogatz_graph(n=n, k=k, p=p, name_suffix=str(i)))

        for n, m in barabasi_albert_params:
            graphs.append(create_barabasi_albert_graph(n=n, m=m, name_suffix=str(i)))

    for graph in graphs:
        save_graph_to_col_file(graph, generated_graphs_directory)
