# -*- coding: utf-8 -*-
"""Module containing main algorithm logic."""

import logging

from networkx import Graph

from algorithm_helper import *
from coloring.partial_color_strategy.color_and_fix import color_and_fix
from coloring.partial_color_strategy.color_indsets import color_indsets
from configuration.algorithm_options_config import optimal_coloring_nodes_threshold
from optimal_coloring import compute_optimal_coloring_dp


class ColoringAlgorithm:

    def __init__(self, color_func, algorithm_name):
        """Creates algorithm coloring graphs using color_func procedure.

        Args:
            color_func (nx.Graph->dict): Function that takes nx.Graph and returns vertex: color dictionary
            algorithm_name (str): Optional algorithm name.
        """

        self._color_graph = color_func
        if algorithm_name is not None:
            self._name = algorithm_name
        else:
            self._name = str(self.color_graph)

    def color_graph(self, graph, partial_coloring=None):
        """Color graph using self._color_graph and ignoring 'partial_coloring' and 'verbose' parameters"""

        return self._color_graph(graph)

    def get_algorithm_name(self):
        return self._name


class PartialColoringAlgorithm:

    def __init__(self, partial_color_strategy, partial_color_strategy_params, algorithm_name):
        self._name = algorithm_name
        self._partial_color_strategy = partial_color_strategy  # function
        self._partial_color_strategy_params = partial_color_strategy_params

    def color_graph(self, graph):
        """Colors graph using vector coloring algorithm.

        Args:
            graph (Graph): Graph to be colored.

        Returns:
            dict: Global vertex-color dictionary indexed from 0 to graph.number_of_nodes()-1.
        """

        if graph.number_of_selfloops() > 0:
            raise Exception('Graph contains self loops')

        partial_coloring = {v: -1 for v in graph.nodes()}

        max_iterations = graph.number_of_nodes() * 2  # is it a good boundary?
        working_graph = graph.copy()

        iteration = 0
        while (working_graph.number_of_nodes() >= 0 and -1 in set(
                partial_coloring.values())) and iteration < max_iterations:
            iteration += 1
            logging.info(
                '\nIteration nr {0} of main loop on graph with {1} nodes and {2} edges...'.format(
                    iteration, working_graph.number_of_nodes(), working_graph.number_of_edges()))

            if working_graph.number_of_nodes() < optimal_coloring_nodes_threshold:
                compute_optimal_coloring_dp(working_graph, partial_coloring, update_graph_and_coloring=True)
                break
            if working_graph.number_of_nodes() > 1 and working_graph.number_of_edges() > 0:
                self._do_partial_color_nonempty_graph_with_edges(working_graph, partial_coloring)
            elif working_graph.number_of_nodes() == 1:
                self._do_color_single_vertex(graph, list(working_graph.nodes)[0], partial_coloring)
                break
            elif working_graph.number_of_edges() == 0:
                self._do_color_no_edges(working_graph, partial_coloring)
                break
            else:
                break

        return partial_coloring

    def _do_partial_color_nonempty_graph_with_edges(self, graph, partial_coloring):
        """ Tries to color at least one vertex of graph. Modifies the graph by deleting nodes that have been colored.

        graph - get modified by coloring some of it's vertices
        partial_coloring - dictionary with partial_coloring of vertices of graph - get updated
        """

        current_number_of_nodes = graph.number_of_nodes()
        while graph.number_of_nodes() == current_number_of_nodes:
            self._partial_color_strategy(
                graph=graph,
                partial_coloring=partial_coloring,
                partial_color_strategy_params=self._partial_color_strategy_params)

    @staticmethod
    def _do_color_single_vertex(graph, vertex, partial_coloring):
        partial_coloring[vertex] = get_lowest_legal_color(graph, vertex, partial_coloring)

    @staticmethod
    def _do_color_no_edges(working_graph, partial_coloring):
        new_color = max(partial_coloring.values()) + 1
        for v in working_graph.nodes():
            partial_coloring[v] = new_color

    def get_algorithm_name(self):
        return self._name


def create_partial_coloring_algorithm(partial_coloring_params, algorithm_name):
    """ Method creates PartialColoringAlgorithm object based on parameters given as dictionary of string -> string
        mappings.

        partial_coloring_params:
            partial_color_strategy,
            partition_strategy_params,
            partial_color_strategy_data_params,
    """

    partial_color_strategy_map = {
        'color_and_fix': color_and_fix.color_all_vertices_at_once,
        'color_indsets': color_indsets.color_by_independent_sets,
    }

    algorithm = PartialColoringAlgorithm(
        partial_color_strategy=lambda graph, partial_coloring, partial_color_strategy_params:
        partial_color_strategy_map[partial_coloring_params['partial_color_strategy']](
            graph, partial_coloring, partial_color_strategy_params),
        partial_color_strategy_params=partial_coloring_params['partial_color_strategy_params'],
        algorithm_name=algorithm_name
    )

    return algorithm
