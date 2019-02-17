from graph_io import *
from results_processing import *

# Test graph creation
tested_graphs = []

tested_graphs.append(read_graph_from_file("dimacs", "DSJC125.1", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "DSJC125.5", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "DSJC125.9", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "DSJC250.1", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "DSJC250.5", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "DSJC250.9", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "DSJC500.1", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "DSJR500.1", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "flat300_20_0", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "flat300_26_0", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "flat300_28_0", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "fpsol2.i.1", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "fpsol2.i.2", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "fpsol2.i.3", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "mulsol.i.1", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "mulsol.i.2", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "mulsol.i.3", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "mulsol.i.4", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "mulsol.i.5", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "zeroin.i.1", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "zeroin.i.2", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "zeroin.i.3", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "games120", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "huck", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "jean", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "anna", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "david", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "le450_5a", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_5b", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_5c", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_5d", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_15a", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_15b", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_15c", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_15d", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_25a", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_25b", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_25c", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "le450_25d", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "miles250", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "miles500", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "miles750", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "miles1000", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "miles1500", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "myciel2", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "myciel3", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "myciel4", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "myciel5", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "myciel6", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "myciel7", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "queen5_5", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen6_6", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen7_7", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen8_8", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen8_12", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen9_9", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen10_10", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen11_11", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen12_12", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen13_13", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen14_14", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen15_15", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "queen16_16", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "r125.1", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "r125.1c", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "r125.5", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "r250.1", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "r250.1c", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "r250.5", starting_index=1))

tested_graphs.append(read_graph_from_file("dimacs", "school1", starting_index=1))
tested_graphs.append(read_graph_from_file("dimacs", "school1_nsh", starting_index=1))

with open("/home/hubert/graph_table", 'w') as outfile:
    results = load_algorithm_run_data_from_file('AggregatedResults')
    for graph in tested_graphs:
        outfile.write("{0} & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\\n".format(graph.name))
