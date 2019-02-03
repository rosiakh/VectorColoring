"""Module containing main algorithm logic."""

import itertools
import logging
import sys

from mosek.fusion import *
from networkx import Graph

import color_all_vertices_at_once
import color_by_independent_sets
import wigderson
from algorithm_helper import *

partial_color_strategy_map = {
    'color_all_vertices_at_once': color_all_vertices_at_once.color_all_vertices_at_once,
    'color_by_independent_sets': color_by_independent_sets.color_by_independent_sets,
}
find_ind_sets_strategy_map = {
    'random_vector_projection': color_by_independent_sets.find_ind_set_by_random_vector_projection,
    'clustering': color_by_independent_sets.find_ind_set_by_clustering,
    None: None,
}
partition_strategy_map = {
    'vector_projection': color_all_vertices_at_once.hyperplanes_partition_strategy,
    'clustering': color_all_vertices_at_once.clustering_partition_strategy,
    'kmeans_clustering': color_all_vertices_at_once.kmeans_clustering_partition_strategy,
    None: None,
}
wigderson_strategy_map = {
    'no_wigderson': wigderson.no_wigderson_strategy,
    'recursive_wigderson': wigderson.recursive_wigderson_strategy,
}


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

    def color_graph(self, g, colors=None, verbose=False):
        """Color graph using self._color_graph and ignoring 'colors' and 'verbose' parameters"""

        return self._color_graph(g)

    def get_algorithm_name(self):
        return self.name


class VectorColoringAlgorithm:

    def __init__(self,
                 partial_color_strategy,
                 wigderson_strategy='no_wigderson',
                 partition_strategy=None,
                 normal_vectors_generation_strategy='random_normal',
                 find_ind_sets_strategy=None,
                 independent_set_extraction_strategy='max_degree_first',
                 sdp_type='nonstrict',
                 alg_name=None):
        """Describe here the interfaces for all those strategies"""

        # TODO: Check for wrong strategy parameters

        init_params = {}

        # TODO: this 'if' is unnecessary but it shows what params are really used in each category
        if partial_color_strategy == 'color_all_vertices_at_once':
            init_params['partition_strategy'] = partition_strategy_map[partition_strategy]
            init_params['independent_set_extraction_strategy'] = independent_set_extraction_strategy
            init_params['normal_vectors_generation_strategy'] = normal_vectors_generation_strategy
        elif partial_color_strategy == 'color_by_independent_sets':
            init_params['find_independent_sets_strategy'] = find_ind_sets_strategy_map[find_ind_sets_strategy]
            init_params['independent_set_extraction_strategy'] = independent_set_extraction_strategy

        self._partially_color_strategy = lambda graph, L, colors: \
            partial_color_strategy_map[partial_color_strategy](graph, L, colors, init_params)

        self._wigderson_strategy = wigderson_strategy_map[wigderson_strategy]

        self._sdp_type = sdp_type

        if alg_name is not None:
            self._name = alg_name
        else:
            self._name = "pcs: " + partial_color_strategy + " ws: " + wigderson_strategy

    def color_graph(self, graph, verbose=False):
        """Colors graph using vector coloring algorithm.

        Args:
            graph (Graph): Graph to be colored.
            verbose (bool): Set verbosity level e.g. for solver.

        Returns:
            dict: Global vertex-color dictionary indexed from 0 to graph.number_of_nodes()-1.
        """

        if graph.number_of_selfloops() > 0:
            raise Exception('Graph contains self loops')

        colors = {v: -1 for v in graph.nodes()}

        logging.info('Starting color_graph procedure on a graph with {0} vertices and {1} edges...'.format(
            graph.number_of_nodes(), graph.number_of_edges()))

        max_iterations = graph.number_of_nodes() * 2  # is it a good boundary?
        working_graph = graph.copy()

        it = 0
        while (working_graph.number_of_nodes() >= 0 and -1 in set(colors.values())) and it < max_iterations:
            it += 1
            logging.info('\nStarting iteration nr {0} of main loop...'.format(it))
            if working_graph.number_of_nodes() > 1 and working_graph.number_of_edges() > 0:
                L = compute_vector_coloring(working_graph, sdp_type=self._sdp_type, verbose=verbose, iteration=it)
                if it == 1:
                    if self._wigderson_strategy(working_graph, colors, L):
                        continue  # Wigderson colored some vertices so we need to recompute vector coloring
                current_nodes = working_graph.number_of_nodes()
                while working_graph.number_of_nodes() == current_nodes:
                    self._partially_color_strategy(working_graph, L, colors)
            elif working_graph.number_of_nodes() == 1:
                colors[list(working_graph.nodes())[0]] = get_lowest_legal_color(graph, list(working_graph.nodes())[0],
                                                                                colors)
                break
            elif working_graph.number_of_edges() == 0:
                new_color = max(colors.values()) + 1
                for v in working_graph.nodes():
                    colors[v] = new_color
                break
            else:
                break

        return colors

    def get_algorithm_name(self):
        return self._name


def compute_vector_coloring(graph, sdp_type, verbose, iteration=-1):
    """Computes sdp_type vector coloring of graph using Cholesky decomposition.

        Args:
            graph (nx.Graph): Graph to be processed.
            sdp_type (string): Non-strict, Strict or Strong coloring.
            verbose (bool): Solver verbosity.
            iteration (int): Number of main algorithm iteration. Used for vector coloring loading or saving.
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

        eps = 1e-7
        for i in range(M.shape[0]):
            M[i, i] = M[i, i] + eps

        M = np.linalg.cholesky(M)

        logging.info('Cholesky factorization computed')
        return M

    def compute_matrix_coloring(graph, sdp_type, verbose):
        """Finds matrix coloring M of graph using Mosek solver.

        Args:
            graph (nx.Graph): Graph to be processed.
            sdp_type (string): Non-strict, Strict or Strong vector coloring.
            verbose (bool): Sets verbosity level of solver.

        Returns:
            2-dim matrix: Matrix coloring of graph G.

        Notes:
            Maybe we can add epsilon to SDP constraints instead of 'solve' parameters?

            For some reason optimal value of alpha is greater than value computed from M below if SDP is solved with big
                tolerance for error

            TODO: strong vector coloring
        """

        logging.info('Computing matrix coloring of graph with {0} nodes and {1} edges...'.format(
            graph.number_of_nodes(), graph.number_of_edges()
        ))

        with Model() as M:

            # Variables
            n = graph.number_of_nodes()
            alpha = M.variable("alpha", Domain.lessThan(0.))
            m = M.variable(Domain.inPSDCone(n))

            # Constraints
            M.constraint("C1", m.diag(), Domain.equalsTo(1.0))
            for i in range(n):
                for j in range(n):
                    if i > j and has_edge_between_ith_and_jth(graph, i, j):
                        if sdp_type == 'strict':
                            M.constraint('C{0}-{1}'.format(i, j), Expr.sub(m.index(i, j), alpha),
                                         Domain.equalsTo(0.))
                        elif sdp_type == 'nonstrict':
                            M.constraint('C{0}-{1}'.format(i, j), Expr.sub(m.index(i, j), alpha),
                                         Domain.lessThan(0.))

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

    # if iteration == 1 and vector_coloring_in_file(graph, strict):
    # L = read_vector_coloring_from_file(graph, strict)
    # else:
    M = compute_matrix_coloring(graph, sdp_type, verbose)
    L = cholesky_factorize(M)
    # if iteration == 1:
    #     save_vector_coloring_to_file(graph, sdp_type, L)

    return L


def compute_optimal_coloring_lp(graph, verbose=False):
    """Computes optimal coloring using linear programming."""

    with Model() as M:

        n = graph.number_of_nodes()

        # Variables
        x = M.variable([n, n], Domain.binary())
        w = M.variable("w", n, Domain.binary())

        # Constraints
        M.constraint('X', Expr.sum(x, 1), Domain.equalsTo(1))

        for i in range(n):
            for j in range(n):
                M.constraint('C{0}-{1}'.format(i, j), Expr.sub(x.index(i, j), w.index(j)),
                             Domain.lessThan(0.0))

        for i in range(n):
            for j in range(n):
                if i > j and has_edge_between_ith_and_jth(graph, i, j):
                    for k in range(n):
                        M.constraint('D{0}-{1}-{2}'.format(i, j, k), Expr.add(x.index(i, k), x.index(j, k)),
                                     Domain.lessThan(1.0))

        # Objective
        M.objective(ObjectiveSense.Minimize, Expr.sum(w))

        # Set solver parameters
        M.setSolverParam("numThreads", 0)

        if verbose:
            M.setLogHandler(sys.stdout)

        M.solve()

        coloring = {}

        for i, v in enumerate(sorted(list(graph.nodes()))):
            for c in range(n):
                if x.index(i, c).level() == 1.0:
                    coloring[v] = c
                    break

    return coloring


def compute_optimal_coloring_dp(graph, verbose=False):
    """Computes optimal coloring using dynamic programming."""

    t_sets = {w: [] for r in range(graph.number_of_nodes() + 1) for w in itertools.combinations(graph.nodes(), r)}
    t = {w: -1 for r in range(graph.number_of_nodes() + 1) for w in itertools.combinations(graph.nodes(), r)}  # set(w)
    t[()] = 0

    for w in t.keys():
        if len(w) == 1:
            t[w] = 1
            t_sets[w] = [w]

    for w in sorted(t.keys(), key=len):
        if len(w) <= 1:
            continue

        min_chi = graph.number_of_nodes()
        min_s = []
        subsets = [s for r in range(1, len(w) + 1) for s in itertools.combinations(w, r)]  # non-empty subsets

        for s in subsets:
            if len(s) == len(w):
                g_s = graph.subgraph(s)
                if g_s.number_of_edges() > 0:
                    continue
            else:
                if t[s] > 1:  # S is not independent
                    continue

            temp = list(set(w) - set(s))
            temp.sort()
            temp = tuple(temp)

            if t[temp] < min_chi:
                min_chi = t[temp]
                min_s = s

        t[w] = min_chi + 1
        t_sets[w] = min_s

    # Now compute the coloring
    ind_sets = []
    vertices = tuple(graph.nodes())
    while vertices:
        i_set = t_sets[vertices]
        vertices = tuple(v for v in vertices if v not in i_set)
        ind_sets.append(i_set)

    coloring = {}
    clr = -1
    for v_set in ind_sets:
        clr += 1
        for v in v_set:
            coloring[v] = clr

    return coloring
