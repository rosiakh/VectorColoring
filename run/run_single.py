# -*- coding: utf-8 -*-
# This file is used for running a chosen algorithm on a single graph. It is invoked when running the program from
# command line or from gui.

from algorithm_runner import run_check_save_on_graphs
from configuration import algorithm_instances_config
from graph.graph_io import *


def run_single(graph_path, algorithm):
    """Runs whole program with a single algorithm on a single graph.

    :param graph_path: path to the file with the graph
    :param algorithm: algorithm to use
    :return: coloring of graph
    """

    logging.basicConfig(format='%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

    (_, filename) = os.path.split(graph_path)
    (graph_name, _) = os.path.splitext(filename)
    graph = read_graph_from_file(path=graph_path)

    graphs = [graph]
    algorithms = [algorithm]

    return run_check_save_on_graphs(graphs, algorithms)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print "Usage: python run_single.py <path_to_graph_file>.col <algorithm>"
        sys.exit()

    if not os.path.exists(sys.argv[1]):
        print "File {0} doesn't exist".format(sys.argv[1])
        sys.exit()

    if not sys.argv[2] in algorithm_instances_config.algorithms_configured:
        print "Algorithm {0} doesn't exist".format(sys.argv[2])
        sys.exit()

    run_single(sys.argv[1], algorithm_instances_config.algorithms_configured[sys.argv[2]])
