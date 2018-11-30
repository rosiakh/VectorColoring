"""Script for running the algorithms on specified graphs.

Usage: python test.py
"""

from algorithm import *
from graph_io import *
from graph_create import *
import time
import json
import logging
import sys


def display_colorings(colorings):
    """Displays results of the coloring algorithms to stdout."""
    for g in colorings:
        for (alg_name, alg_coloring) in colorings[g]:
            print '\n'
            if (not check_if_coloring_legal(g, alg_coloring)):
                print 'Coloring obtained by \n\t{0} \non \n\t{1} \nis not legal'.format(alg_name, g.name)
            else:
                print 'Coloring obtained by \n\t{0} \non \n\t{1} \nis legal and uses {2} colors'.format(
                    alg_name, g.name, str(len(set(alg_coloring.values()))))


# Logging configuration
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Test graph creation
graphs = []
graphs.append(read_graph_from_file("dimacs", "DSJC125.1", starting_index=1))
# graphs.append(create_k_cycle(20, 90))

# Test algorithm creation
algorithms = []
algorithms.append(VectorColoringAlgorithm(
    partially_color_graph_by_random_hyperplane_partition_strategy,
    no_wigderson_strategy))
algorithms.append(ColoringAlgorithm(
    lambda g: nx.algorithms.coloring.greedy_color(g, strategy='independent_set'), 'greedy_independent_set'))
algorithms.append(ColoringAlgorithm(
    lambda g: nx.algorithms.coloring.greedy_color(g, strategy='DSATUR'), 'dsatur'))

# Run algorithms to obtain colorings
colorings = {}
for g in graphs:
    colorings[g] = []
    for alg in algorithms:
        colorings[g].append((alg.get_algorithm_name(), alg.color_graph(g, verbose=False)))

logging.shutdown()

# Check if colorings are legal
for g in colorings:
    for (alg_name, alg_coloring) in colorings[g]:
        if (not check_if_coloring_legal(g, alg_coloring)):
            raise Exception('Coloring obtained by {0} on {1} is not legal'.format(alg_name, g.name))

display_colorings(colorings)
