"""
Configuration of algorithm instances to be used in the program.

Available options for algorithm creation:

partial_color_strategy:
        'color_all_vertices_at_once',
        'color_by_independent_sets'

independent_set_extraction_strategy:
        'max_degree_first',
        'min_vertex_cover',
        'arora_kms',
        'arora_kms_prim'

sdp_type:
        'nonstrict',
        'strict',
        'strong'

wigderson_strategy:
        'no_wigderson',
        'recursive_wigderson'

find_independent_sets_strategy:
        'random_vector_projection',
        'clustering'

partition_strategy: (only for partial_color_strategy='color_all_vertices_at_once')
        'hyperplane_partition',
        'clustering'

normal_vectors_generation_strategy: (only for 'partition_strategy':'hyperplane_partition')
        'random_normal',
        'orthonormal'
"""

import algorithm_options_config
from coloring.algorithm import *
from coloring.optimal_coloring import compute_optimal_coloring_lp, compute_optimal_coloring_dp

algorithms_configured = {}

algorithms_configured['Color and fix: clustering'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_and_fix',
        'partial_color_strategy_params': {
            'partition_strategy': 'clustering',
            'independent_set_extraction_strategy': 'max_degree_first',
            'deterministic': True,
            'nr_of_vector_colorings_to_try': 1,  # For now it makes no sense to set it other than 1
            'partition_strategy_params': {
                'sdp_type': 'nonstrict',
                'nr_of_cluster_sizes_to_check': 5,
                'cluster_size_lower_factor': 0.9,  # Makes no sense to set it much lower than 1.0
                'cluster_size_upper_factor': 1.5,
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='Color and fix: clustering')

algorithms_configured['Color and fix: orthonormal hyperplanes'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_and_fix',
        'partial_color_strategy_params': {
            'partition_strategy': 'hyperplane_partition',
            'independent_set_extraction_strategy': 'max_degree_first',
            'deterministic': False,
            'nr_of_vector_colorings_to_try': 1,  # For now it makes no sense to set it other than 1
            'partition_strategy_params': {
                'normal_vectors_generation_strategy': 'orthonormal',
                'sdp_type': 'nonstrict',
                'nr_of_partitions_per_vc': 25,
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='Color and fix: orthonormal hyperplanes')

algorithms_configured['Color and fix: random hyperplanes'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_and_fix',
        'partial_color_strategy_params': {
            'partition_strategy': 'hyperplane_partition',
            'independent_set_extraction_strategy': 'max_degree_first',
            'deterministic': False,
            'nr_of_vector_colorings_to_try': 1,  # For now it makes no sense to set it other than 1
            'partition_strategy_params': {
                'normal_vectors_generation_strategy': 'random_normal',
                'sdp_type': 'nonstrict',
                'nr_of_partitions_per_vc': 25,
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='Color and fix: random hyperplanes')

algorithms_configured['IndSets strategy: clustering'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'clustering',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'independent_set_extraction_strategy': 'max_degree_first',
                'sdp_type': 'nonstrict',
                'nr_of_cluster_sizes_to_check': 15,
                'cluster_size_lower_factor': 0.9,  # Makes no sense to set it much lower than 1.0
                'cluster_size_upper_factor': 1.5,
            },
        },
    },
    algorithm_name='IndSets strategy: clustering')

algorithms_configured['IndSets strategy: random vector projection'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'random_vector_projection',
            'deterministic': False,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'independent_set_extraction_strategy': 'max_degree_first',
                'sdp_type': 'nonstrict',
                'nr_of_random_vector_sets_to_try': 30,
                'max_nr_of_random_vectors_without_change': 15,
                'c_param_lower_factor': 0.3,
                'c_param_upper_factor': 1.5,
                'nr_of_c_params_tried_per_random_vector': 5,
            },
        },
    },
    algorithm_name='IndSets strategy: random vector projection')

algorithms_configured['IndSets strategy: directed vector coloring'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'dummy_vector_capturing',
            'deterministic': False,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {

            },
        },
    },
    algorithm_name='IndSets strategy: directed vector coloring')

algorithms_configured['Greedy: Independent Set'] = ColoringAlgorithm(
    color_func=lambda graph: nx.algorithms.coloring.greedy_color(graph, strategy='independent_set'),
    algorithm_name='Greedy: Independent Set')

algorithms_configured['Greedy: DSATUR'] = ColoringAlgorithm(
    color_func=lambda graph: nx.algorithms.coloring.greedy_color(graph, strategy='DSATUR'),
    algorithm_name='Greedy: DSATUR')

algorithms_configured['Optimal: linear programming'] = ColoringAlgorithm(
    color_func=lambda graph: compute_optimal_coloring_lp(graph),
    algorithm_name='Optimal: linear programming')

algorithms_configured['Optimal: dynamic programming'] = ColoringAlgorithm(
    color_func=lambda graph: compute_optimal_coloring_dp(graph),
    algorithm_name='Optimal: dynamic programming')

gui_algorithms_sorted = [
    ColoringAlgorithm(
        color_func=lambda graph: nx.algorithms.coloring.greedy_color(graph, strategy='independent_set'),
        algorithm_name='Greedy: Independent Set'),

    ColoringAlgorithm(
        color_func=lambda graph: nx.algorithms.coloring.greedy_color(graph, strategy='DSATUR'),
        algorithm_name='Greedy: DSATUR'),

    ColoringAlgorithm(
        color_func=lambda graph: compute_optimal_coloring_lp(graph),
        algorithm_name='Optimal: linear programming'),

    ColoringAlgorithm(
        color_func=lambda graph: compute_optimal_coloring_dp(graph),
        algorithm_name='Optimal: dynamic programming'),

    create_partial_coloring_algorithm(
        partial_coloring_params={
            'partial_color_strategy': 'color_all_vertices_at_once',
            'partial_color_strategy_params': {
                'partition_strategy': 'hyperplane_partition',
                'normal_vectors_generation_strategy': 'orthonormal',
                'independent_set_extraction_strategy': 'max_degree_first',
                'deterministic': False,
                'sdp_type': 'nonstrict',
            },
            'partial_color_strategy_data_params': algorithm_options_config.default_color_and_fix_params,
        },
        algorithm_name='Color and fix: orthonormal hyperplanes'),

    create_partial_coloring_algorithm(
        partial_coloring_params={
            'partial_color_strategy': 'color_all_vertices_at_once',
            'partial_color_strategy_params': {
                'partition_strategy': 'hyperplane_partition',
                'normal_vectors_generation_strategy': 'random_normal',
                'independent_set_extraction_strategy': 'max_degree_first',
                'deterministic': False,
                'sdp_type': 'nonstrict',
            },
            'partial_color_strategy_data_params': algorithm_options_config.default_color_and_fix_params,
        },
        algorithm_name='Color and fix: random hyperplanes'),

    create_partial_coloring_algorithm(
        partial_coloring_params={
            'partial_color_strategy': 'color_all_vertices_at_once',
            'partial_color_strategy_params': {
                'partition_strategy': 'clustering',
                'independent_set_extraction_strategy': 'max_degree_first',
                'deterministic': True,
                'sdp_type': 'nonstrict',
            },
            'partial_color_strategy_data_params': algorithm_options_config.default_color_and_fix_params,
        },
        algorithm_name='Color and fix: clustering'),

    create_partial_coloring_algorithm(
        partial_coloring_params={
            'partial_color_strategy': 'color_by_independent_sets',
            'partial_color_strategy_params': {
                'find_independent_sets_strategy': 'random_vector_projection',
                'independent_set_extraction_strategy': 'max_degree_first',
                'deterministic': False,
                'sdp_type': 'nonstrict',
            },
            'partial_color_strategy_data_params': algorithm_options_config.default_color_indsets_params,
        },
        algorithm_name='IndSets strategy: random vector projection'),

    create_partial_coloring_algorithm(
        partial_coloring_params={
            'partial_color_strategy': 'color_by_independent_sets',
            'partial_color_strategy_params': {
                'find_independent_sets_strategy': 'clustering',
                'independent_set_extraction_strategy': 'max_degree_first',
                'deterministic': True,
                'sdp_type': 'nonstrict',
            },
            'partial_color_strategy_data_params': algorithm_options_config.default_color_indsets_params,
        },
        algorithm_name='IndSets strategy: clustering'),
]
