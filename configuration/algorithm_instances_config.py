"""Configuration of algorithm instances to be used in the program.

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

algorithms_configured['Color and fix -clustering'] = create_partial_coloring_algorithm(
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
    algorithm_name='Color and fix -clustering')

algorithms_configured['Color and fix -orthonormal hyperplanes'] = create_partial_coloring_algorithm(
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
    algorithm_name='Color and fix -orthonormal hyperplanes')

algorithms_configured['Color and fix -random hyperplanes'] = create_partial_coloring_algorithm(
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
    algorithm_name='Color and fix -random hyperplanes')

algorithms_configured['IndSets strategy -clustering'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'clustering',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'independent_set_extraction_strategy': 'max_degree_first',
                'sdp_type': 'nonstrict',
                'nr_of_cluster_sizes_to_check': 1,
                'cluster_size_lower_factor': 1.0,  # Makes no sense to set it much lower than 1.0
                'cluster_size_upper_factor': 1.2,
            },
        },
    },
    algorithm_name='IndSets strategy -clustering')

algorithms_configured[
    'IndSets strategy -random vector projection -best-many-trials_25x25'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'random_vector_projection',
            'deterministic': False,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'c_adaptation_strategy': 'linspace',
                'c_adaptation_strategy_params': {
                    'c_param_lower_factor': 0.2,
                    'c_param_upper_factor': 3.0,
                    'nr_of_c_params_tried_per_random_vector': 25,
                },
                'independent_set_extraction_strategy': 'max_degree_first',
                'sdp_type': 'nonstrict',
                'nr_of_random_vector_sets_to_try': 25,
                'max_nr_of_random_vectors_without_change': 15,
            },
        },
    },
    algorithm_name='IndSets strategy -random vector projection -best-many-trials_25x25')

algorithms_configured['IndSets strategy -random vector projection -linspace'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'random_vector_projection',
            'deterministic': False,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'c_adaptation_strategy': 'linspace',
                'c_adaptation_strategy_params': {
                    'c_param_lower_factor': 0.5,
                    'c_param_upper_factor': 1.5,
                    'nr_of_c_params_tried_per_random_vector': 10,
                },
                'independent_set_extraction_strategy': 'max_degree_first',
                'sdp_type': 'nonstrict',
                'nr_of_random_vector_sets_to_try': 1,
                'max_nr_of_random_vectors_without_change': 15,
            },
        },
    },
    algorithm_name='IndSets strategy -random vector projection -linspace')

algorithms_configured['IndSets strategy -random vector projection -ratio -low'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'random_vector_projection',
            'deterministic': False,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'c_adaptation_strategy': 'ratio',
                'c_adaptation_strategy_params': {
                    'ratio_lower_bound': 0.4,
                    'c_lower_bound': 0.1,
                    'c_decrease_ratio': 0.95,
                },
                'independent_set_extraction_strategy': 'max_degree_first',
                'sdp_type': 'nonstrict',
                'nr_of_random_vector_sets_to_try': 1,
                'max_nr_of_random_vectors_without_change': 15,
            },
        },
    },
    algorithm_name='IndSets strategy -random vector projection -ratio -low')

algorithms_configured['IndSets strategy -random vector projection -ratio -high'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'random_vector_projection',
            'deterministic': False,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'c_adaptation_strategy': 'ratio',
                'c_adaptation_strategy_params': {
                    'ratio_lower_bound': 0.7,
                    'c_lower_bound': 0.1,
                    'c_decrease_ratio': 0.7,
                },
                'independent_set_extraction_strategy': 'max_degree_first',
                'sdp_type': 'nonstrict',
                'nr_of_random_vector_sets_to_try': 1,
                'max_nr_of_random_vectors_without_change': 15,
            },
        },
    },
    algorithm_name='IndSets strategy -random vector projection -ratio -high')

algorithms_configured[
    'IndSets strategy -dummy vector coloring -projection -linspace -1_c_param_per_vector'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'dummy_vector_strategy',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'beta_factor_strategy': {
                    'name': 'uniform',
                    'factor': 1.5,
                },
                'c_adaptation_strategy': 'linspace',
                'c_adaptation_strategy_params': {
                    'c_param_lower_factor': 1.0,
                    'c_param_upper_factor': 1.0,
                    'nr_of_c_params_tried_per_random_vector': 1,
                    'initial_c_percentile': 35,
                },
                'alpha_upper_bound': 0.0,
                'find_almost_indsets_strategy': 'projection',
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='IndSets strategy -dummy vector coloring -projection -linspace -1_c_param_per_vector')

algorithms_configured[
    'IndSets strategy -dummy vector coloring -projection -linspace -10_c_param_per_vector'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'dummy_vector_strategy',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'beta_factor_strategy': {
                    'name': 'uniform',
                    'factor': 1.5,
                },
                'c_adaptation_strategy': 'linspace',
                'c_adaptation_strategy_params': {
                    'c_param_lower_factor': 0.5,
                    'c_param_upper_factor': 1.5,
                    'nr_of_c_params_tried_per_random_vector': 10,
                    'initial_c_percentile': 35,
                },
                'alpha_upper_bound': 0.0,
                'find_almost_indsets_strategy': 'projection',
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='IndSets strategy -dummy vector coloring -projection -linspace -10_c_param_per_vector')

algorithms_configured[
    'IndSets strategy -dummy vector coloring -projection -ratio -fast'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'dummy_vector_strategy',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'beta_factor_strategy': {
                    'name': 'uniform',
                    'factor': 1.5,
                },
                'c_adaptation_strategy': 'ratio',
                'c_adaptation_strategy_params': {
                    'ratio_lower_bound': 0.6,
                    'c_lower_bound': 0.1,
                    'c_decrease_ratio': 0.7,
                    'initial_c_percentile': 10,
                },
                'alpha_upper_bound': 0.0,
                'find_almost_indsets_strategy': 'projection',
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='IndSets strategy -dummy vector coloring -projection -ratio -fast')

algorithms_configured[
    'IndSets strategy -dummy vector coloring -projection -ratio -slow'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'dummy_vector_strategy',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'beta_factor_strategy': {
                    'name': 'uniform',
                    'factor': 1.5,
                },
                'c_adaptation_strategy': 'ratio',
                'c_adaptation_strategy_params': {
                    'ratio_lower_bound': 0.3,
                    'c_lower_bound': 0.1,
                    'c_decrease_ratio': 0.95,
                    'initial_c_percentile': 10,
                },
                'alpha_upper_bound': 0.0,
                'find_almost_indsets_strategy': 'projection',
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='IndSets strategy -dummy vector coloring -projection -ratio -slow')

algorithms_configured[
    'IndSets strategy -dummy vector coloring -greedy -all-vertices'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'dummy_vector_strategy',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'beta_factor_strategy': {
                    'name': 'uniform',
                    'factor': 1.5,
                },
                'greedy_continue_strategy': 'all-vertices',
                'greedy_continue_strategy_params': {},
                'alpha_upper_bound': 0.0,
                'find_almost_indsets_strategy': 'greedy',
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='IndSets strategy -dummy vector coloring -greedy -all-vertices')

algorithms_configured[
    'IndSets strategy -dummy vector coloring -greedy -all-vertices ver2'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'dummy_vector_strategy',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'beta_factor_strategy': {
                    'name': 'uniform',
                    'factor': 1.0,
                },
                'greedy_continue_strategy': 'all-vertices',
                'greedy_continue_strategy_params': {},
                'alpha_upper_bound': 1.0,
                'find_almost_indsets_strategy': 'greedy',
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='IndSets strategy -dummy vector coloring -greedy -all-vertices ver2')

algorithms_configured[
    'IndSets strategy -dummy vector coloring -greedy -first-edge'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'dummy_vector_strategy',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'beta_factor_strategy': {
                    'name': 'uniform',
                    'factor': 1.5,
                },
                'greedy_continue_strategy': 'first-edge',
                'greedy_continue_strategy_params': {},
                'alpha_upper_bound': 0.0,
                'find_almost_indsets_strategy': 'greedy',
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='IndSets strategy -dummy vector coloring -greedy -first-edge')

algorithms_configured[
    'IndSets strategy -dummy vector coloring -greedy -ratio -high'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'dummy_vector_strategy',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'beta_factor_strategy': {
                    'name': 'uniform',
                    'factor': 1.5,
                },
                'greedy_continue_strategy': 'ratio',
                'greedy_continue_strategy_params': {
                    'lower_bound_nr_of_nodes': 10,
                    'ratio_lower_bound': 0.8,
                },
                'alpha_upper_bound': 0.0,
                'find_almost_indsets_strategy': 'greedy',
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='IndSets strategy -dummy vector coloring -greedy -ratio -high')

algorithms_configured[
    'IndSets strategy -dummy vector coloring -greedy -ratio -low'] = create_partial_coloring_algorithm(
    partial_coloring_params={
        'partial_color_strategy': 'color_indsets',
        'partial_color_strategy_params': {
            'find_independent_sets_strategy': 'dummy_vector_strategy',
            'deterministic': True,
            'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
            'find_indsets_strategy_params': {
                'beta_factor_strategy': {
                    'name': 'uniform',
                    'factor': 1.5,
                },
                'greedy_continue_strategy': 'ratio',
                'greedy_continue_strategy_params': {
                    'lower_bound_nr_of_nodes': 10,
                    'ratio_lower_bound': 0.5,
                },
                'alpha_upper_bound': 0.0,
                'find_almost_indsets_strategy': 'greedy',
                'independent_set_extraction_strategy': 'max_degree_first',
            },
        },
    },
    algorithm_name='IndSets strategy -dummy vector coloring -greedy -ratio -low')

algorithms_configured['Greedy -Independent Set'] = ColoringAlgorithm(
    color_func=lambda graph: nx.algorithms.coloring.greedy_color(graph, strategy='independent_set'),
    algorithm_name='Greedy -Independent Set')

algorithms_configured['Greedy -DSATUR'] = ColoringAlgorithm(
    color_func=lambda graph: nx.algorithms.coloring.greedy_color(graph, strategy='DSATUR'),
    algorithm_name='Greedy -DSATUR')

algorithms_configured['Optimal: linear programming'] = ColoringAlgorithm(
    color_func=lambda graph: compute_optimal_coloring_lp(graph),
    algorithm_name='Optimal: linear programming')

algorithms_configured['Optimal: dynamic programming'] = ColoringAlgorithm(
    color_func=lambda graph: compute_optimal_coloring_dp(graph),
    algorithm_name='Optimal: dynamic programming')

gui_algorithms_sorted = [
    ColoringAlgorithm(
        color_func=lambda graph: nx.algorithms.coloring.greedy_color(graph, strategy='independent_set'),
        algorithm_name='Greedy -Independent Set'),

    ColoringAlgorithm(
        color_func=lambda graph: nx.algorithms.coloring.greedy_color(graph, strategy='DSATUR'),
        algorithm_name='Greedy -DSATUR'),

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
        algorithm_name='Color and fix -orthonormal hyperplanes'),

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
        algorithm_name='Color and fix -random hyperplanes'),

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
        algorithm_name='Color and fix -clustering'),

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
        algorithm_name='IndSets strategy -random vector projection'),

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
        algorithm_name='IndSets strategy -clustering'),
]
