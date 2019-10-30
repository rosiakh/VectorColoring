"""Configuration of paths used in the program."""

import os

run_seed = ''  # A String used to differentiate between runs of algorithm, set at run's start.

generated_graphs_directory = '../resources/graph_instances/generated/'
test_graphs_directory = '../resources/graph_instances/test_instances/'

base_directory = '/home/hubert/VectorColoring/'
if not os.path.exists(base_directory):
    os.makedirs(base_directory)


def results_directory(seed=None):
    if seed is None:
        seed = run_seed
    directory = current_run_directory(seed) + 'Results/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def drawings_directory(seed=None):
    if seed is None:
        seed = run_seed
    directory = current_run_directory(seed) + 'Drawings/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def vector_colorings_directory(seed=None):
    if seed is None:
        seed = run_seed
    directory = current_run_directory(seed) + 'VectorColorings/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def logs_directory(seed=None):
    if seed is None:
        seed = run_seed
    directory = current_run_directory(seed) + 'Logs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def current_run_directory(seed=None):
    if seed is None:
        seed = run_seed
    directory = base_directory + seed + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


chromatic_numbers_path = '../resources/chromatic_numbers.txt'
latex_document_top_path = '../resources/latex_document_top'
latex_result_path = base_directory + 'latex_result'
algorithm_info_filename = "algorithm_info.txt"
