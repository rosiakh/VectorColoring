import os

import numpy as np

from configuration import paths_config


def save_vector_coloring_to_file(graph, sdp_type, vector_coloring):
    filename = get_vector_coloring_filename(graph, sdp_type)
    np.savetxt(filename, vector_coloring)


def vector_coloring_in_file(graph, sdp_type):
    filename = get_vector_coloring_filename(graph, sdp_type)

    return os.path.exists(filename)


def read_vector_coloring_from_file(graph, sdp_type):
    filename = get_vector_coloring_filename(graph, sdp_type)

    return np.loadtxt(filename)


def get_vector_coloring_filename(graph, sdp_type):
    filename = paths_config.current_run_directory() + '/_' + graph.name + '_sdp_type=' + str(sdp_type) + '_vc'

    return filename
