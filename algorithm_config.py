"""
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

normal_vectors_generation_strategy: (only for partition_strategy='hyperplane_partition')
        'random_normal',
        'orthonormal'


"""

from algorithm import *
from results_processing import *

algorithms = {}

algorithms['orthonormal_hyperplane_partition'] = VectorColoringAlgorithm(
    partial_color_strategy='color_all_vertices_at_once',
    partition_strategy='hyperplane_partition',
    normal_vectors_generation_strategy='orthonormal',
    independent_set_extraction_strategy='max_degree_first',
    wigderson_strategy='no_wigderson',
    sdp_type='nonstrict',
    alg_name='orthonormal hyperplane partition',
    deterministic=False
)

algorithms['clustering_all_vertices'] = VectorColoringAlgorithm(
    partial_color_strategy='color_all_vertices_at_once',
    partition_strategy='clustering',
    independent_set_extraction_strategy='max_degree_first',
    wigderson_strategy='no_wigderson',
    sdp_type='nonstrict',
    alg_name='clustering all vertices',
    deterministic=True
)

algorithms['random_hyperplane_partition'] = VectorColoringAlgorithm(
    partial_color_strategy='color_all_vertices_at_once',
    partition_strategy='hyperplane_partition',
    normal_vectors_generation_strategy='random_normal',
    independent_set_extraction_strategy='max_degree_first',
    wigderson_strategy='no_wigderson',
    sdp_type='nonstrict',
    alg_name='random hyperplane partition',
    deterministic=False
)

algorithms['random_vector_projection'] = VectorColoringAlgorithm(
    partial_color_strategy='color_by_independent_sets',
    find_independent_sets_strategy='random_vector_projection',
    independent_set_extraction_strategy='max_degree_first',
    wigderson_strategy='no_wigderson',
    sdp_type='nonstrict',
    alg_name='random vector projection',
    deterministic=False
)

algorithms['clustering_independent_sets'] = VectorColoringAlgorithm(
    partial_color_strategy='color_by_independent_sets',
    find_independent_sets_strategy='clustering',
    independent_set_extraction_strategy='max_degree_first',
    wigderson_strategy='no_wigderson',
    sdp_type='nonstrict',
    alg_name='clustering independent sets',
    deterministic=True
)

algorithms['greedy_independent_set'] = ColoringAlgorithm(
    lambda graph: nx.algorithms.coloring.greedy_color(graph, strategy='independent_set'), 'greedy_independent_set')

algorithms['dsatur'] = ColoringAlgorithm(
    lambda graph: nx.algorithms.coloring.greedy_color(graph, strategy='DSATUR'), 'dsatur')

algorithms['optimal_coloring_linear_programming'] = ColoringAlgorithm(
    lambda graph: compute_optimal_coloring_lp(graph, verbose=True), 'optimal_lp')

algorithms['optimal_coloring_dynamic_programming'] = ColoringAlgorithm(
    lambda graph: compute_optimal_coloring_dp(graph, verbose=True), 'optimal_dp')
