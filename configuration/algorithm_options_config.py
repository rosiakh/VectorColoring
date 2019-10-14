# -*- coding: utf-8 -*-
"""Configuration of some of the options of algorithms."""

sdp_strong_threshold = -1  # for smaller graphs solver always solves strong sdp
use_previous_sdp_result = False  # if set to True other algorithms may use sdp results of other algorithms in the same run
solver_name = 'mosek'  # other possibilities: cvxopt; TODO: maybe try picos as well
solver_verbose = False
solver_output = 'stdout'
general_verbosity = False
parallel = False
nr_of_parallel_jobs = 1
optimal_coloring_nodes_threshold = 14

# this config contains parameters related to algorithm's data;
# for config related to behavior see 'partition_strategy_params'
default_color_indsets_params = {
    'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
    'nr_of_random_vector_sets_to_try': 30,
    'max_nr_of_random_vectors_without_change': 15,
    'c_param_lower_factor': 0.3,
    'c_param_upper_factor': 1.5,
    'nr_of_c_params_tried_per_random_vector': 5,
    'nr_of_cluster_sizes_to_check': 15,
    'cluster_size_lower_factor': 0.9,  # Makes no sense to set it much lower than 1.0
    'cluster_size_upper_factor': 1.5,
}

default_color_and_fix_params = {
    'nr_of_vector_colorings_to_try': 1,  # For now it makes no sense to set it other than 1
    'nr_of_partitions_per_vc': 25,
    'nr_of_cluster_sizes_to_check': 5,
    'cluster_size_lower_factor': 0.9,  # Makes no sense to set it much lower than 1.0
    'cluster_size_upper_factor': 1.5,
}
