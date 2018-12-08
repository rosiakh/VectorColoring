"""Module for loading, saving and displaying graphs."""

import os

import matplotlib.pyplot as plt
import networkx as nx
import pylab
import numpy as np


def read_graph_from_file(folder_name, graph_name, starting_index=0):
    """Creates graph from DIMACS-format file.

    Args:
        graph_name (str): Name of the file to read, without directory and extension.
        starting_index (int): Index of first vertex.

    Returns:
        Graph: networkx.Graph object created from given file with nodes indexed from 0.
    """

    G = nx.Graph()

    filename = 'graph_instances/' + folder_name + '/' + graph_name + '.col'
    with open(filename) as f:
        for line in f:
            l = line.split()
            if l[0] == 'p':
                nr_of_nodes = int(l[2])
                G.add_nodes_from(range(starting_index, starting_index + nr_of_nodes))
            elif l[0] == 'e':
                try:
                    e1 = int(l[1])
                    e2 = int(l[2])
                    G.add_edge(e1, e2)
                except ValueError:
                    G.add_edge(l[1], l[2])

    # G = nx.relabel_nodes(G, lambda x: x - 1, copy=False)
    G.name = graph_name

    return G


def draw_graph(G, colors, toConsole=True, toImage=False, filename='graph'):
    """Draws graph and saves image to file.

    Args:
        G (Graph): Graph to be drawn.
        colors (dict): Global vertex-color dictionary.
        filename (str): File to save.
    """

    output_dir = '/home/hubert/Desktop/vc-graphs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for v in colors.keys():
    # colors[v] = list(set(colors.values())).index(colors[v])

    if colors:
        colors_list = [v for k, v in sorted(colors.items())]
        # print 'colors list:', colors_list
        # print 'colors set:', set(colors_list)
        print 'number of colors used:', len(set(colors_list))
        # print 'bin colors: ', [bin(colors[v]) for v in range(0, G.number_of_nodes() + 0)]
    else:
        colors_list = [1] * G.number_of_nodes()

    if toConsole:
        print 'name:', G.name
        print 'number of nodes:', G.number_of_nodes()
        print 'number of edges:', G.number_of_edges()
        # print 'edges:', G.edges()

    if toImage:
        node_labels = {v: str(v + 0) for v in range(G.number_of_nodes())}
        fig = pylab.figure(figsize=(11, 8))

        nx.draw(G, pos=nx.circular_layout(G), with_labels=True, node_color=colors_list, cmap=plt.cm.Spectral,
                labels=node_labels)

        # hide axis
        fig.gca().axes.get_xaxis().set_ticks([])
        fig.gca().axes.get_yaxis().set_ticks([])

        pylab.savefig('{0}/{1}.png'.format(output_dir, filename), format="PNG")
        # pylab.show()


def display_graph_stats(G):
    """Displays some graphs stats.

    Args:
        G (Graph): Graph to display.
    """

    degrees = dict(G.degree()).values()
    n = G.number_of_nodes()
    max_degree = max(degrees)
    min_degree = min(degrees)
    avg_degree = np.average(degrees)

    print ' current graph stats:'
    print '     number of vertices:', n
    print '     max degree:', max_degree
    print '     avg degree:', avg_degree
    print '     min degree:', min_degree
    print '     number of edges:', G.number_of_edges()
    print '     edge density:', float(G.number_of_edges()) / float((n * (n - 1) / 2))