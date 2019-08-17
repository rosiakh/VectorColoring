from multiprocessing import Process, Lock, Manager

from color_indsets import *
from configuration import algorithm_options_config


def color_by_independent_sets_parallel(graph, partial_coloring, partial_color_strategy_params):
    """This strategy finds one or more independent set finding it one list of sets at a time."""

    logging.info('Looking for independent sets...')

    vector_coloring = compute_vector_coloring(graph, partial_color_strategy_params['sdp_type'])

    # TODO: strategy of determining how many sets to get at once from find_ind_set_strategy

    best_ind_sets = None

    manager = Manager()
    shmem_ind_sets = manager.list()
    lock = Lock()
    processes = []

    iterations = 1 if partial_color_strategy_params['deterministic'] else \
        partial_color_strategy_params[
            'nr_of_times_restarting_ind_set_strategy'] / algorithm_options_config.nr_of_parallel_jobs
    nr_jobs = 1 if partial_color_strategy_params[
        'deterministic'] else algorithm_options_config.nr_of_parallel_jobs

    if iterations > 1 and graph.number_of_nodes() < 100:
        iterations = max(int(float(graph.number_of_nodes()) / float(100) * iterations), 1)

    for _ in range(iterations):
        for _ in range(nr_jobs):
            processes.append(
                Process(
                    target=find_independent_sets_strategy_map[
                        partial_color_strategy_params['find_independent_sets_strategy']],
                    args=(
                        graph, vector_coloring, partial_color_strategy_params, get_nr_of_sets_at_once(graph),
                        shmem_ind_sets,
                        lock)))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        processes = []

    best_ind_sets = better_ind_sets_parallel(graph, shmem_ind_sets, best_ind_sets)

    update_coloring_and_graph(graph, partial_coloring, best_ind_sets)
    logging.debug('Found independent sets (maybe identical) of sizes: ' + str([len(s) for s in best_ind_sets]))


def get_nr_of_sets_at_once(graph):
    """Determines maximal number of independent sets found for one vector coloring."""

    # if nx.classes.density(graph) > 0.5 and graph.number_of_nodes() > 100:
    #     return max(1, int(math.floor((nx.classes.density(graph) + 0.5) * (graph.number_of_nodes() - 50) / 25)))

    return 1


def better_ind_sets_parallel(graph, ind_sets1, ind_sets2):
    """ind_sets2 are current best that get replaced """

    best = ind_sets2
    for i in range(len(ind_sets1)):
        if is_better_ind_sets(graph, ind_sets1[i], best):
            best = ind_sets1[i]

    return best
