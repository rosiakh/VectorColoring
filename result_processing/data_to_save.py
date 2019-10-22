import networkx as nx


class DataToSave:
    """Class represent all data that is saved for one graph x one algorithm (possibly multiple runs) based on
        graph (networkx.Graph) and run_result (RunResults) objects.
    """

    def __init__(self, graph, run_result):
        self.graph_name = graph.name
        self.graph_family = graph.family
        self.graph_nr_of_vertices = graph.number_of_nodes()
        self.graph_density = nx.classes.density(graph)
        self.avg_nr_of_colors = run_result.average_nr_of_colors
        self.min_nr_of_colors = len(set(run_result.best_coloring.values()))
        self.best_coloring = run_result.best_coloring
        self.algorithm_name = run_result.algorithm.get_algorithm_name()
        self.avg_time = run_result.average_time
