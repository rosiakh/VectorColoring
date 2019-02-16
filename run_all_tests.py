from datetime import datetime
from timeit import default_timer as timer

from algorithm import *
from graph_io import *
from results_processing import *

# Logging configuration
logging.basicConfig(format='%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

# Test graph creation
graphs = []

# graphs.append(create_erdos_renyi_graph(n=10, p=0.5))
# graphs.append(create_barabasi_albert_graph(n=20, m=20))
# graphs.append(create_watts_strogatz_graph(n=30, k=6, p=0.6))
# graphs.append(create_k_cycle(k=2, n=20))
# graphs.append(nx.random_partition_graph(sizes=[x for x in range(15, 23)], p_in=0.9, p_out=0.2))

# graphs.append(read_graph_from_file('other', 'grotzsch', starting_index=0))

# graphs.append(read_graph_from_file("dimacs", "DSJC125.1", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "DSJC125.5", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "DSJC125.9", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "DSJC250.1", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "DSJC250.5", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "DSJC250.9", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "DSJC500.1", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "DSJC500.5", starting_index=1))  vertices >
# graphs.append(read_graph_from_file("dimacs", "DSJC500.9", starting_index=1))  vertices >

# graphs.append(read_graph_from_file("dimacs", "DSJR500.1", starting_index=1))  vertices >
# graphs.append(read_graph_from_file("dimacs", "DSJR500.1c", starting_index=1))  vertices >
# graphs.append(read_graph_from_file("dimacs", "DSJR500.5", starting_index=1))  vertices >

# graphs.append(read_graph_from_file("dimacs", "flat300_20_0", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "flat300_26_0", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "flat300_28_0", starting_index=1))  done
#
# graphs.append(read_graph_from_file("dimacs", "fpsol2.i.1", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "fpsol2.i.2", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "fpsol2.i.3", starting_index=1))  done

# graphs.append(read_graph_from_file("dimacs", "inithx.i.1", starting_index=1))  vertices >
# graphs.append(read_graph_from_file("dimacs", "inithx.i.2", starting_index=1))  vertices >
# graphs.append(read_graph_from_file("dimacs", "inithx.i.3", starting_index=1))  vertices >

# graphs.append(read_graph_from_file("dimacs", "mulsol.i.1", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "mulsol.i.2", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "mulsol.i.3", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "mulsol.i.4", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "mulsol.i.5", starting_index=1))  do later

# graphs.append(read_graph_from_file("dimacs", "zeroin.i.1", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "zeroin.i.2", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "zeroin.i.3", starting_index=1))  done
#
# graphs.append(read_graph_from_file("dimacs", "games120", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "homer", starting_index=1))  vertices >
# graphs.append(read_graph_from_file("dimacs", "huck", starting_index=1)) done
# graphs.append(read_graph_from_file("dimacs", "jean", starting_index=1)) done
# graphs.append(read_graph_from_file("dimacs", "anna", starting_index=1)) done
# graphs.append(read_graph_from_file("dimacs", "david", starting_index=1)) done
#
# graphs.append(read_graph_from_file("dimacs", "le450_5a", starting_index=1)) done
# graphs.append(read_graph_from_file("dimacs", "le450_5b", starting_index=1)) done
# graphs.append(read_graph_from_file("dimacs", "le450_5c", starting_index=1)) done
# graphs.append(read_graph_from_file("dimacs", "le450_5d", starting_index=1)) done
# graphs.append(read_graph_from_file("dimacs", "le450_15a", starting_index=1)) done
# graphs.append(read_graph_from_file("dimacs", "le450_15b", starting_index=1)) do later
# graphs.append(read_graph_from_file("dimacs", "le450_15c", starting_index=1)) do later
# graphs.append(read_graph_from_file("dimacs", "le450_15d", starting_index=1)) do later
# graphs.append(read_graph_from_file("dimacs", "le450_25a", starting_index=1)) do later
# graphs.append(read_graph_from_file("dimacs", "le450_25b", starting_index=1)) do later
# graphs.append(read_graph_from_file("dimacs", "le450_25c", starting_index=1)) do later
# graphs.append(read_graph_from_file("dimacs", "le450_25d", starting_index=1)) do later

# graphs.append(read_graph_from_file("dimacs", "miles250", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "miles500", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "miles750", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "miles1000", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "miles1500", starting_index=1))  done

# graphs.append(read_graph_from_file("dimacs", "myciel2", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "myciel3", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "myciel4", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "myciel5", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "myciel6", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "myciel7", starting_index=1))  done
#
# graphs.append(read_graph_from_file("dimacs", "queen5_5", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "queen6_6", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "queen7_7", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "queen8_8", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "queen8_12", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "queen9_9", starting_index=1))  done
# graphs.append(read_graph_from_file("dimacs", "queen10_10", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "queen11_11", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "queen12_12", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "queen13_13", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "queen14_14", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "queen15_15", starting_index=1))  do later
# graphs.append(read_graph_from_file("dimacs", "queen16_16", starting_index=1))  do later

graphs.append(read_graph_from_file("dimacs", "r125.1", starting_index=1))
graphs.append(read_graph_from_file("dimacs", "r125.1c", starting_index=1))
graphs.append(read_graph_from_file("dimacs", "r125.5", starting_index=1))
graphs.append(read_graph_from_file("dimacs", "r250.1", starting_index=1))
graphs.append(read_graph_from_file("dimacs", "r250.1c", starting_index=1))
graphs.append(read_graph_from_file("dimacs", "r250.5", starting_index=1))

graphs.append(read_graph_from_file("dimacs", "school1", starting_index=1))
graphs.append(read_graph_from_file("dimacs", "school1_nsh", starting_index=1))

algorithms = []

algorithms.append(VectorColoringAlgorithm(
    partial_color_strategy='color_all_vertices_at_once',
    partition_strategy='hyperplane_partition',
    normal_vectors_generation_strategy='orthonormal',
    independent_set_extraction_strategy='max_degree_first',
    wigderson_strategy='no_wigderson',
    sdp_type='nonstrict',
    alg_name='orthonormal hyperplane partition',
    deterministic=False
))

algorithms.append(VectorColoringAlgorithm(
    partial_color_strategy='color_all_vertices_at_once',
    partition_strategy='clustering',
    independent_set_extraction_strategy='max_degree_first',
    wigderson_strategy='no_wigderson',
    sdp_type='nonstrict',
    alg_name='clustering all vertices',
    deterministic=True
))

algorithms.append(VectorColoringAlgorithm(
    partial_color_strategy='color_all_vertices_at_once',
    partition_strategy='hyperplane_partition',
    normal_vectors_generation_strategy='random_normal',
    independent_set_extraction_strategy='max_degree_first',
    wigderson_strategy='no_wigderson',
    sdp_type='nonstrict',
    alg_name='random hyperplane partition',
    deterministic=False
))

algorithms.append(VectorColoringAlgorithm(
    partial_color_strategy='color_by_independent_sets',
    find_independent_sets_strategy='random_vector_projection',
    independent_set_extraction_strategy='max_degree_first',
    wigderson_strategy='no_wigderson',
    sdp_type='nonstrict',
    alg_name='random vector projection',
    deterministic=False
))

algorithms.append(VectorColoringAlgorithm(
    partial_color_strategy='color_by_independent_sets',
    find_independent_sets_strategy='clustering',
    independent_set_extraction_strategy='max_degree_first',
    wigderson_strategy='no_wigderson',
    sdp_type='nonstrict',
    alg_name='clustering independent sets',
    deterministic=True
))


algorithms.append(ColoringAlgorithm(
    lambda g: nx.algorithms.coloring.greedy_color(g, strategy='independent_set'), 'greedy_independent_set'))

algorithms.append(ColoringAlgorithm(
    lambda g: nx.algorithms.coloring.greedy_color(g, strategy='DSATUR'), 'dsatur'))

# Run algorithms to obtain colorings
repetitions_per_graph = 1
algorithms_results = {}  # Dictionary - graph: list of RunResults (one result per algorithm)
config.run_seed = datetime.now().strftime("%m-%d_%H-%M-%S")
for graph_counter, graph in enumerate(graphs):
    algorithms_results[graph] = []
    for alg_counter, alg in enumerate(algorithms):
        logging.info("\nComputing graph: {0} ({2}/{3}), algorithm: {1} ({4}/{5}) ...\n".format(
            graph.name, alg.get_algorithm_name(), graph_counter + 1, len(graphs), alg_counter + 1, len(algorithms)))
        nrs_of_colors = []
        times = []
        graph_colorings = []
        for iteration in range(repetitions_per_graph):
            start = timer()
            coloring = alg.color_graph(graph, verbose=True)
            end = timer()
            times.append(end - start)
            graph_colorings.append(coloring)

        results = RunResults()
        results.graph = graph
        results.algorithm = alg
        results.average_time = np.mean(times)
        results.best_coloring = min(graph_colorings, key=lambda coloring: len(set(coloring.values())))
        results.average_nr_of_colors = np.mean([len(set(coloring.values())) for coloring in graph_colorings])
        results.repetitions = repetitions_per_graph

        algorithms_results[graph].append(results)
        logging.info("Done graph: {0}, algorithm: {1}, colors: {2}, time: {3:6.2f} s ...\n".format(
            graph.name, alg.get_algorithm_name(), len(set(results.best_coloring.values())), results.average_time))
    save_graph_run_data_to_file(algorithms_results[graph], graph)

logging.shutdown()

# Check if colorings are legal
for graph in algorithms_results:
    for results in algorithms_results[graph]:
        if not check_if_coloring_legal(graph, results.best_coloring):
            raise Exception('Coloring obtained by {0} on {1} is not legal'.format(results.algorithm.name, graph.name))
