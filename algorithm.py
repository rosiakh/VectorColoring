"""Module containing main algorithm logic."""

import math

import logging
import cvxpy
import networkx as nx
import networkx.algorithms.approximation
import numpy as np
from networkx import Graph
from scipy.stats import norm
import sys
import mosek
from mosek.fusion import *
from mosek import callbackcode, iinfitem, dinfitem, liinfitem


class ColoringAlgorithm:

    def __init__(self, color_func, alg_name=None):
        """Creates algorithm coloring graphs using color_func procedure.

        Args:
            color_func (nx.Graph->dict): Function that takes nx.Graph and returns vertex: color dictionary
            alg_name (str): Optional algorithm name.
        """

        self._color_graph = color_func
        if alg_name is not None:
            self.name = alg_name
        else:
            self.name = str(self.color_graph)

    def color_graph(self, G, colors=None, verbose=False):
        """Color graph using self._color_graph and ignoring 'colors' and 'verbose' parameters"""

        return self._color_graph(G)

    def get_algorithm_name(self):
        return self.name


class VectorColoringAlgorithm:

    def __init__(self,
                 partially_color_strategy=None,
                 wigderson_strategy=None,
                 alg_name=None):
        """Describe here the interfaces for all those strategies"""

        self._partially_color_strategy = partially_color_strategy
        self._wigderson_strategy = wigderson_strategy

        if alg_name is not None:
            self.name = alg_name
        else:
            self.name = "pcs=" + self._partially_color_strategy.__name__ + " ws=" + self._wigderson_strategy.__name__

    def color_graph(self, G, colors=None, verbose=False, precolors=None):
        """Colors graph using vector coloring algorithm.

        Args:
            G (Graph): Graph to be colored.
            colors (dict): Partial coloring of G
            verbose (bool): Set verbosity level e.g. for solver.
            precolors (dict): Full legal precoloring of graph G.

        Returns:
            dict: Global vertex-color dictionary indexed from 0 to G.number_of_nodes()-1.
        """

        if G.number_of_selfloops() > 0:
            raise Exception('Graph contains selfloops')

        if colors is None:
            colors = {v: -1 for v in G.nodes()}
        else:
            pass  # TODO: Delete colored vertices and edges; change color values to start from 0 continuously

        max_iterations = G.number_of_nodes() * 2  # is it a good boundary?
        current_G = G.copy()

        logging.info('Starting color_graph procedure on a graph with {0} vertices and {1} edges...'.format(
            G.number_of_nodes(), G.number_of_edges()))

        it = 0
        while current_G.number_of_nodes() >= 0 and it < max_iterations:
            it += 1
            logging.info('\n')
            logging.info('Starting iteration nr {0} of main loop...'.format(it))
            if current_G.number_of_nodes() > 1 and current_G.number_of_edges() > 0:
                M = compute_matrix_coloring(current_G, precolors, verbose=verbose)
                L = compute_vector_coloring(M)
                self._partially_color_strategy(current_G, L, colors)
            elif current_G.number_of_nodes() == 1:
                colors[list(current_G.nodes())[0]] = max(colors.values()) + 1
                break
            elif current_G.number_of_edges() == 0:
                new_color = max(colors.values()) + 1
                for v in current_G.nodes():
                    colors[v] = new_color
                break
            else:
                break

        return colors

    def get_algorithm_name(self):
        return self.name


# END CLASS


def compute_vector_coloring(M):
    """Computes vector coloring on the basis of matrix coloring using Cholesky decomposition.

        Args:
            M (2-dim matrix): Matrix-coloring of the graph.

        Returns:
              2-dim matrix: Rows of this matrix are vectors of computed vector coloring.
        """

    def cholesky_factorize(M):
        """Returns L such that M = LL^T.

            According to https://en.wikipedia.org/wiki/Cholesky_decomposition#Proof_for_positive_semi-definite_matrices
                if L is positive semi-definite then we can turn it into positive definite by adding eps*I.

            We can also perform LDL' decomposition and set L = LD^(1/2) - it works in Matlab even though M is singular.

            It sometimes returns an error if M was computed with big tolerance for error.

            Args:
                M (2-dim matrix): Positive semidefinite matrix to be factorized.

            Returns:
                L (2-dim matrix): Cholesky factorization of M such that M = LL^T.
            """

        logging.info('Starting Cholesky factorization...')

        eps = 10e-8
        L = M

        for i in range(L.shape[0]):
            L[i, i] = L[i, i] + eps  # TODO: Should I normalize the output vector coloring?

        L = np.linalg.cholesky(L)

        # Probably don't need these lines
        for i in range(L.shape[0]):
            for j in range(i + 1, L.shape[0]):
                L[i, j] = 0

        logging.info('Cholesky factorization computed')
        return L

    return cholesky_factorize(M)


def check_if_coloring_legal(G, colors, partial=False):
    """Checks if given coloring is a legal vertex coloring.

        Args:
            G (graph): Colored graph.
            colors (dict): Global vertex-color dictionary.
            partial (bool): If True, than we check for legal partial colloring, i.e. vertices with both endpoints=-1
                are legal

        Returns:
            bool: True iff coloring of G given by colors is legal (partial) vertex coloring.
        """

    for i, j in G.edges():
        if colors[i] == colors[j]:
            if colors[i] == -1 and partial:
                continue
            else:
                return False

    return True


def compute_matrix_coloring(G, precolors, strict=False, verbose=False):
    """Finds matrix coloring M of graph G using CVXPY.

    Maybe we can add epsilon to SDP constraints instead of 'solve' parameters?

    For some reason optimal value of alpha is greater than value computed from M below if SDP is solved with big
        tolerance for error

    Args:
        G (Graph): Graph to be processed.
        strict (bool): Are we looking for strict vector coloring.
        verbose (bool): Sets verbosity level of solver.
        precolors (dict): Full legal coloring of graph G used to obtain good starting point for solver.

    Returns:
        2-dim matrix: Matrix coloring of graph G.
    """

    logging.info('Computing matrix coloring of graph with {0} nodes and {1} edges...'.format(
        G.number_of_nodes(), G.number_of_edges()
    ))

    def has_edge_between_ith_and_jth(G, i, j):
        """Checks if there is an edge in G between i-th vertex and j-th vertex after sorting them.

            Graph may have vertex called 'i' that isn't it's i-th vertex in sorted order (e.g. when some vertices have been
                removed from the graph). This function checks if there is an edge between i-th and j-th vertex in sorted
                order, so i-th and j-th vertex exist as long as those numbers are less than G.number_of_nodes()

            Args:
                G (Graph): Graph to be processed
                i (int): Number between 0 and G.number_of_nodes()-1
                j (int): Number between 0 and G.number_of_nodes()-1

            Returns:
                bool: True iff there is an edge in G between i-th vertex and j-th vertex after sorting them.
            """

        return G.has_edge(sorted(list(G.nodes()))[i], sorted(list(G.nodes()))[j])

    with Model() as M:

        # Variables
        n = G.number_of_nodes()
        alpha = M.variable("alpha", Domain.lessThan(0.))
        m = M.variable(Domain.inPSDCone(n))

        # Constraints
        M.constraint("C1", m.diag(), Domain.equalsTo(1.0))
        for i in range(n):
            for j in range(n):
                if i > j and has_edge_between_ith_and_jth(G, i, j):
                    M.constraint('C{0}{1}{2}'.format(i, j, i * j), Expr.sub(m.index(i, j), alpha), Domain.lessThan(0.))

        # Objective
        M.objective(ObjectiveSense.Minimize, alpha)

        # Set solver parameters
        M.setSolverParam("numThreads", 0)

        if verbose:
            M.setLogHandler(sys.stdout)

        M.solve()

        alpha_opt = alpha.level()[0]
        level = m.level()
        result = [[level[j * n + i] for i in range(n)] for j in range(n)]
        result = np.array(result)

    logging.info('Found matrix {0}-coloring'.format(1 - 1 / alpha_opt))

    return result


# partially color graph strategies and helper

def partially_color_graph_by_hyperplane_partition_strategy(G, L, colors):
    """Colors some of the vertices of the graph given it's vector coloring.

    Vertices are colored using partition of the space by random hyperplanes. When parameters are set correctly then
        with high probability at least half of the vertices should be colored i.e. their values in 'colors' dictionary
        should be set to some non-negative number.

    Args:
        G (Graph): Graph to be colored. It is modified so that only not colored vertices are left.
        L (2-dim matrix): Rows of L are vector coloring of G. Nth row is assigned to nth vertex which doesn't have to
            be vertex 'n'.
        colors (dict): Global vertex-color dictionary.
    """

    def check_if_better_partition(best_subgraph_illegal_edges, current_subgraph_illegal_edges):
        """Returns true if current partition is better than best partition.

        Hyperplane partitions can be assesed using different strategies. The best found so far is to first compare
            number of illegal edges and then number of colors used in the partition.

        Args:
            best_subgraph_illegal_edges (networkx.Graph): Subgraph of illegal edges of best partition so far.
            current_subgraph_illegal_edges (networkx.Graph): Subgraph of illegal edges of current partition.

        Returns:
            bool: True iff current partition is better than previous best partition.
        """

        # return subgraph_illegal_edges.number_of_nodes() - subgraph_illegal_edges.number_of_edges() \
        # > best_subgraph_illegal_edges.number_of_nodes() - best_subgraph_illegal_edges.number_of_edges()

        return subgraph_illegal_edges.number_of_edges() < best_subgraph_illegal_edges.number_of_edges() \
               or (subgraph_illegal_edges.number_of_edges() == best_subgraph_illegal_edges.number_of_edges()
                   and len(set(temp_colors.values())) < len(set(best_temp_colors.values())))

    degrees = dict(G.degree()).values()
    n = G.number_of_nodes()
    max_degree = max(degrees)

    k = find_number_of_vector_colors_from_vector_coloring(G, L)
    nr_of_hyperplanes = 2 + int(math.ceil(math.log(max_degree, k)))  # Need to verify that it is optimal.

    max_iterations = 1e2
    for iteration in range(int(max_iterations)):
        hyperplanes_sides = {v: 0 for v in range(0, n)}
        temp_colors = {v: -1 for v in G.nodes()}
        for i in range(nr_of_hyperplanes):
            r = np.random.normal(0, 1, n)
            x = np.sign(np.dot(L, r))
            for v in range(0, n):
                if x[v] >= 0:
                    hyperplanes_sides[v] = hyperplanes_sides[v] * 2 + 1
                else:
                    hyperplanes_sides[v] = hyperplanes_sides[v] * 2

        min_available_color = max(colors.values()) + 1
        for i, v in enumerate(
                sorted(G.nodes())):  # I assume here that nodes are given in the same order as in rows of L.
            temp_colors[v] = min_available_color
            temp_colors[v] = temp_colors[v] + hyperplanes_sides[i]

        # Remove colors from one endpoint of each illegal edge.
        illegal_edges = [(i, j) for (i, j) in G.edges() if
                         temp_colors[i] == temp_colors[j] and temp_colors[i] != -1]
        subgraph_illegal_edges = nx.Graph()
        subgraph_illegal_edges.add_edges_from(illegal_edges)

        if iteration == 0:
            best_subgraph_illegal_edges = subgraph_illegal_edges
            best_temp_colors = temp_colors.copy()
        if check_if_better_partition(best_subgraph_illegal_edges, subgraph_illegal_edges):
            best_subgraph_illegal_edges = subgraph_illegal_edges
            best_temp_colors = temp_colors.copy()

    for v in best_temp_colors.keys():
        colors[v] = best_temp_colors[v]

    # We need to remove some vertices of best_subgraph_illegal_edges, so that what is left is legal coloring
    # Legally colored graph is temp_G.
    vertex_cover = nx.algorithms.approximation.min_weighted_vertex_cover(best_subgraph_illegal_edges)
    temp_G = G.copy()
    G.remove_nodes_from([n for n in G.nodes() if n not in vertex_cover])
    for v in vertex_cover:
        colors[v] = -1
    temp_G.remove_nodes_from(G.nodes())
    if not check_if_coloring_legal(temp_G, colors, partial=True):
        raise Exception('coloring is not legal after some hyperplane partition')


def partially_color_graph_by_vector_projection_strategy(G, L, colors):
    """Colors some of the vertices of the graph given it's vector coloring.

        Vertices are colored using partition of the space by vector projections. When parameters are set correctly then
            with high probability at least half of the vertices should be colored i.e. their values in 'colors' dictionary
            should be set to some non-negative number.

        In

        Args:
            G (Graph): Graph to be colored. It is modified so that only not colored vertices are left.
            L (2-dim matrix): Rows of L are vector coloring of G -
                nth row is assigned to nth vertex which doesn't have to be vertex 'n'.
            colors (dict): Global vertex-color dictionary.
            M (2-dim matrix): Matrix coloring of G.
            iterations (int): Number of random vectors (also potentially ind. sets) considered
        """

    def compute_c_opt_parameter(G, L):
        """Computes optimal c parameter according to KMS.

        Args:
            G (nx.Graph): Graph
            L (2-dim array): Vector coloring of G
        """

        max_degree = max(dict(G.degree()).values())
        k = find_number_of_vector_colors_from_vector_coloring(G, L)
        temp = (2 * (k - 2) * math.log(max_degree)) / k
        if temp >= 0:
            c = math.sqrt(temp)
        else:
            c = 0.0

        return c

    def get_nodes_sorted_by_degree(G):
        """Returns list of nodes sorted by degree in descending order."""

        return [item[0] for item in
                sorted(list(G.degree(G.nodes())), key=lambda item: item[1], reverse=True)]

    def compute_theoretical_lower_bound_for_exp_diff():
        """Computes theoretical lower bound for expected difference between number of nodes and edges in found independent set."""

        max_degree = max(dict(G.degree()).values())
        k = find_number_of_vector_colors_from_vector_coloring(G, L)
        c = compute_c_opt_parameter(G, L)
        try:
            a = math.sqrt(2 * (k - 1) / (k - 2))
            lower_bound = max(n * norm.sf(c) - n * max_degree * norm.sf(a * c) / 2, n * norm.sf(c) / 2)
        except:
            lower_bound = -1

        return lower_bound

    logging.info('Looking for independent set using vector projection strategy...')

    nr_of_c_values = 10
    n = G.number_of_nodes()
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(G.nodes()))}
    c_opt = compute_c_opt_parameter(G, L)
    iterations = int(min(max(n ** 2, 1e3), 1e3))

    r = np.random.normal(0, 1, n)
    x = np.dot(L, r)
    best_subgraph_nodes = [inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c_opt]
    best_subgraph_edges = {(i, j) for i, j in G.edges() if
                           (i in best_subgraph_nodes and j in best_subgraph_nodes)}

    logging.debug('Vector projections to consider: {0} x {1}'.format(nr_of_c_values, iterations))

    for c in np.linspace(c_opt / 2, 3 * c_opt / 2, nr_of_c_values):
        it = 0
        while it < iterations:
            it += 1
            r = np.random.normal(0, 1, n)
            x = np.dot(L, r)
            initial_subgraph_nodes = [inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c]
            initial_subgraph_edges = {(i, j) for i, j in G.edges() if
                                      (i in initial_subgraph_nodes and j in initial_subgraph_nodes)}

            current_difference = len(initial_subgraph_nodes) - len(initial_subgraph_edges)
            if current_difference > len(best_subgraph_nodes) - len(best_subgraph_edges):
                best_subgraph_nodes = initial_subgraph_nodes
                best_subgraph_edges = initial_subgraph_edges

    subgraph = nx.Graph()
    subgraph.add_nodes_from(best_subgraph_nodes)
    subgraph.add_edges_from(best_subgraph_edges)

    if subgraph.number_of_nodes() == 0:
        logging.info('Didn\'t manage to find any independent set')
        return

    # remove one node from each edge to obtain an independent set
    nodes_by_degree = get_nodes_sorted_by_degree(subgraph)

    while nodes_by_degree:
        if subgraph.degree(nodes_by_degree[0]) == 0:
            break
        subgraph.remove_node(nodes_by_degree[0])
        nodes_by_degree = get_nodes_sorted_by_degree(subgraph)

    # All nodes have degree 0 so they are independent set. Color it using next free color and remove from G.
    if nodes_by_degree:
        min_available_color = max(colors.values()) + 1
        for v in subgraph.nodes():
            colors[v] = min_available_color
        G.remove_nodes_from(subgraph.nodes())

    logging.info('Found independent set of size {0}'.format(subgraph.number_of_nodes()))


def find_number_of_vector_colors_from_vector_coloring(G, L):
    """Given vector coloring find number of 'vector-colors' used i.e. smallest 'k' such that L is vector k-coloring of G.

        Vector coloring is obtained from matrix coloring computed using SDP optimalization and Cholesky factorization.
            Both of those processes may introduce error and return number of 'vector-colors' greater than vector chromatic
            number.

        Args:
            G (Graph): Graph of which L is vector coloring.
            L (2-dim matrix): Rows of L are vector coloring of G. Nth row is assigned to nth vertex which doesn't
                have to be vertex 'n'.

        Returns:
            int: smallest 'k' such that L is vector k-coloring of G.
        """

    M = np.dot(L, np.transpose(L))

    vertices_mapping = {v: i for i, v in enumerate(sorted(G.nodes()))}
    cells = []
    for i, j in G.edges():
        cells.append(M[vertices_mapping[i], vertices_mapping[j]])

    return 1 - 1 / max(cells)


# wigderson strategies and helper

def max_ind_set_wigderson_strategy(G, colors, threshold_degree):
    """Colors some of the nodes using Wigderson technique.

        Args:
            G (graph): Graph to be processed.
            colors (dict): Global vertex-color dictionary.
            threshold_degree (int): Lower boundary for degree of processed nodes of G.
        """

    print '\n starting Wigderson algorithm:'
    print '     threshold deg:', threshold_degree
    max_degree = max(dict(G.degree()).values())
    iterations = 0
    while max_degree > threshold_degree:
        iterations += 1
        print '\n iteration', iterations
        print ' max deg:', max_degree

        # Find any vertex with degree equal to max_degree.
        max_vertex = 0
        for v in dict(G.degree()):
            if G.degree()[v] == max_degree:
                max_vertex = v
                break

        neighbors_subgraph = G.subgraph(G.neighbors(max_vertex))

        # Find large independent set in neighbors subgraph using approximate maximum independent set algorithm.
        # Can we find this large independent set using our algorithm recursively?
        min_available_color = max(colors.values()) + 1
        max_ind_set = nx.algorithms.approximation.maximum_independent_set(neighbors_subgraph)
        # max_ind_set = nx.maximal_independent_set(neighbors_subgraph)
        for v in max_ind_set:
            colors[v] = min_available_color

        # Remove nodes that have just been colored
        G.remove_nodes_from(max_ind_set)
        max_degree = max(dict(G.degree()).values())

    return


def recursive_wigderson_strategy(G, colors, threshold_degree):
    """Colors some of the nodes using Wigderson technique.

    Args:
        G (graph): Graph to be processed.
        colors (dict): Global vertex-color dictionary.
        threshold_degree (int): Lower boundary for degree of processed nodes of G.
    """

    print '\n starting Wigderson algorithm:'
    print '     threshold deg:', threshold_degree
    max_degree = max(dict(G.degree()).values())
    iterations = 0
    while max_degree > threshold_degree:
        iterations += 1
        print '\n iteration', iterations
        print ' max deg:', max_degree

        # Find any vertex with degree equal to max_degree.
        max_vertex = 0
        for v in dict(G.degree()):
            if G.degree()[v] == max_degree:
                max_vertex = v
                break

        neighbors_subgraph = G.subgraph(G.neighbors(max_vertex))
        neighbors_colors, k = self.color_graph(neighbors_subgraph, colors)
        for v in neighbors_subgraph.nodes():
            colors[v] = neighbors_colors[v]

        # Remove nodes that have just been colored
        G.remove_nodes_from(neighbors_subgraph.nodes())
        max_degree = max(dict(G.degree()).values())

    return


def no_wigderson_strategy(G, colors, threshold_degree):
    pass
