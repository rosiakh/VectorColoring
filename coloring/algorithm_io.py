"""Functions used for saving and loading from file a vector coloring. Currently not used anywhere in the program.
Probably requires update."""

import os

import numpy as np

from configuration import paths_config


def save_vector_coloring_to_file(graph, sdp_type, vector_coloring):
    """
    :param graph: (nx.Graph)
    :param sdp_type: type of semidefinite programming problem
    :param vector_coloring: (2-dim matrix) rows are vectors of vector coloring
    """

    filename = get_vector_coloring_filename(graph, sdp_type)
    np.savetxt(filename, vector_coloring)


def vector_coloring_in_file(graph, sdp_type):
    """
    :param graph: (nx.Graph)
    :param sdp_type: type of semidefinite programming problem
    :return: True iff given vector coloring type results for given graph already exists
    """

    filename = get_vector_coloring_filename(graph, sdp_type)

    return os.path.exists(filename)


def read_vector_coloring_from_file(graph, sdp_type):
    """
    :param graph: (nx.Graph)
    :param sdp_type: type of semidefinite programming problem
    :return: vector coloring read from file
    """

    filename = get_vector_coloring_filename(graph, sdp_type)

    return np.loadtxt(filename)


def get_vector_coloring_filename(graph, sdp_type):
    """
    :param graph: (nx.Graph)
    :param sdp_type: type of semidefinite programming problem
    :return:
    """

    filename = paths_config.current_run_directory() + '/_' + graph.name + '_sdp_type=' + str(sdp_type) + '_vc'

    return filename
