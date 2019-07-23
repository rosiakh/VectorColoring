from result_processing import *


def display_results_on_console(run_seed=None):
    graph_results = load_algorithm_run_data_from_seed(run_seed)

    for graph_name in graph_results:  # graph_results - a list of dictionaries (one dict per algorithm)
        print 'Graph: {0}'.format(graph_name)
        for results in graph_results[graph_name]:
            print '\talgorithm: {0:40}, colors used: {1}'.format(
                results['algorithm_name'], results['min_nr_of_colors'])
