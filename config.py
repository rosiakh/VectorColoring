import os

base_directory = '/home/hubert/VectorColoring/'
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

run_seed = ''  # A String used to differentiate between runs of algorithm, set at run's start.


def vector_colorings_directory():
    directory = current_run_directory() + 'VectorColorings/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def logs_directory():
    directory = current_run_directory() + 'Logs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def current_run_directory():
    directory = base_directory + run_seed + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


color_by_independent_sets_params = {
    'nr_of_times_restarting_ind_set_strategy': 1,
    'nr_of_random_vectors_tried': 150,
    'max_nr_of_random_vectors_without_change': 100,
    'c_param_lower_factor': 0.2,
    'c_param_upper_factor': 1.2,
    'nr_of_c_params_tried_per_random_vector': 10,
    'nr_of_cluster_sizes_to_check': 100,
    'cluster_size_lower_factor': 0.3,
    'cluster_size_upper_factor': 2.0,
}

color_all_vertices_at_once_params = {
    'nr_of_partitions_to_try': 100,
    'nr_of_cluster_sizes_to_check': 100,
    'cluster_size_lower_factor': 0.3,
    'cluster_size_upper_factor': 2.0,
}
