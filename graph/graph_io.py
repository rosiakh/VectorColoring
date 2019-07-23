"""Module for loading, saving and displaying graphs."""

import logging
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pylab

from configuration import paths_config


def read_graph_from_directory_recursively(path):
    """ Returns list of all graphs in a given directory and recursively in it's subdirectories.

    r = root, d = directories, f = files
    """

    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.col' in file:
                files.append(os.path.join(r, file))

    graphs = []
    for f in files:
        graphs.append(read_graph_from_file(path=f))

    return graphs


def read_graphs_from_directory(path):
    """ Returns list of all graphs in a given directory but not in subdirectories. """

    from os import listdir
    from os.path import isfile, join
    graph_files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and '.col' in f]

    graphs = []
    for f in graph_files:
        graphs.append(read_graph_from_file(path=f))

    return graphs


def read_graph_from_file(path):
    """Creates graph from DIMACS-format file.

    Args:
        graph_name (str): Name of the file to read, without directory and extension.

    Returns:
        Graph: networkx.Graph object created from given file with nodes indexed from 0.
    """

    graph = nx.Graph()

    (_, filename) = os.path.split(path)
    (graph_name, _) = os.path.splitext(filename)

    starting_index = find_starting_node_index(path)
    with open(path) as f:
        for line in f:
            l = line.split()
            if l[0] == 'p':
                nr_of_nodes = int(l[2])
                graph.add_nodes_from(range(starting_index, starting_index + nr_of_nodes))
            elif l[0] == 'e':
                try:
                    e1 = int(l[1])
                    e2 = int(l[2])
                    graph.add_edge(e1, e2)
                except ValueError:
                    raise Exception('Wrong vertex number in col file: {0}'.format(path))

    graph.name = graph_name
    graph.family = graph.name
    graph.starting_index = starting_index

    return graph


def draw_graph(graph, coloring, to_console=True, to_image=False, filename='graph'):
    """Draws graph and saves image to file.

    Args:
        graph (Graph): Graph to be drawn.
        coloring (dict): Global vertex-color dictionary.
        filename (str): File to save.
    """

    output_dir = paths_config.drawings_directory()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for v in coloring.keys():
    # coloring[v] = list(set(coloring.values())).index(coloring[v])

    if coloring:
        colors_list = [v for k, v in sorted(coloring.items())]
        print 'number of colors used:', len(set(colors_list))
    else:
        colors_list = [1] * graph.number_of_nodes()

    if to_console:
        print 'name:', graph.name
        print 'number of nodes:', graph.number_of_nodes()
        print 'number of edges:', graph.number_of_edges()

    if to_image:
        node_labels = {v: str(v + 0) for v in range(graph.number_of_nodes())}
        logging.getLogger("matplotlib.backends._backend_tk").setLevel(logging.CRITICAL)
        fig = pylab.figure(figsize=(10, 7))

        nx.draw(graph, pos=nx.circular_layout(graph), with_labels=True, node_color=colors_list, cmap=plt.cm.Spectral,
                labels=node_labels)

        # hide axis
        fig.gca().axes.get_xaxis().set_ticks([])
        fig.gca().axes.get_yaxis().set_ticks([])

        pylab.savefig('{0}/{1}.png'.format(output_dir, filename), format="PNG")
        pylab.show()


def find_starting_node_index(path):
    starting_index = sys.maxint
    with open(path) as f:
        for line in f:
            l = line.split()
            if l[0] == 'e':
                starting_index = min(starting_index, int(l[1]), int(l[2]))
    return starting_index


def display_graph_stats(graph):
    """Displays some graphs stats.

    Args:
        graph (Graph): Graph to display.
    """

    degrees = dict(graph.degree()).values()
    n = graph.number_of_nodes()
    max_degree = max(degrees)
    min_degree = min(degrees)
    avg_degree = np.average(degrees)

    print ' current graph stats:'
    print '     number of vertices:', n
    print '     max degree:', max_degree
    print '     avg degree:', avg_degree
    print '     min degree:', min_degree
    print '     number of edges:', graph.number_of_edges()
    print '     edge density:', float(graph.number_of_edges()) / float((n * (n - 1) / 2))


def save_graph_to_col_file(graph, path):
    filename = path if path.endswith("/") else path + "/"
    filename = filename + graph.name + '.col'

    with open(filename, 'w') as outfile:
        for e1, e2 in graph.edges():
            outfile.write("e {0} {1}\n".format(e1, e2))
