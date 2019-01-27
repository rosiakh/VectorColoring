"""Module containing main algorithm logic."""

import sys

from mosek.fusion import *
from networkx import Graph

from hyperplane_partition import *
from vector_projection import *
from wigderson import *
import itertools

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
                 partial_color_strategy,
                 wigderson_strategy,
                 find_ind_sets_strategy=None,
                 partition_strategy=None,
                 alg_name=None):
        """Describe here the interfaces for all those strategies"""

        # Assign partial color strategy
        if partial_color_strategy == 'hyperplane_partition':
            if partition_strategy == 'random':
                self._partially_color_strategy = lambda g, L, colors: partially_color_graph_by_hyperplane_partition(
                    g,
                    L,
                    colors,
                    partition_strategy=random_partition_strategy)
            elif partition_strategy == 'orthogonal':
                self._partially_color_strategy = lambda g, L, colors: partially_color_graph_by_hyperplane_partition(
                    g,
                    L,
                    colors,
                    partition_strategy=orthogonal_partition_strategy)
            elif partition_strategy == 'clustering':
                self._partially_color_strategy = lambda g, L, colors: partially_color_graph_by_hyperplane_partition(
                    g,
                    L,
                    colors,
                    partition_strategy=clustering_partition_strategy)
            elif partition_strategy == 'kmeans_clustering':
                self._partially_color_strategy = lambda g, L, colors: partially_color_graph_by_hyperplane_partition(
                    g,
                    L,
                    colors,
                    partition_strategy=kmeans_clustering_partition_strategy)
            else:
                raise Exception('Unknown partition strategy')
        elif partial_color_strategy == 'vector_projection':
            if find_ind_sets_strategy == 'random_vector_projection':
                self._partially_color_strategy = lambda g, L, colors: partially_color_graph_by_vector_projections(
                    g,
                    L,
                    colors,
                    find_ind_sets_strategy=find_ind_set_by_random_vector_projection
                )
            elif find_ind_sets_strategy == 'random_vector_projection_kms':
                self._partially_color_strategy = lambda g, L, colors: partially_color_graph_by_vector_projections(
                    g,
                    L,
                    colors,
                    find_ind_sets_strategy=find_ind_set_by_random_vector_projection_kms
                )
            elif find_ind_sets_strategy == 'random_vector_projection_kms_prim':
                self._partially_color_strategy = lambda g, L, colors: partially_color_graph_by_vector_projections(
                    g,
                    L,
                    colors,
                    find_ind_sets_strategy=find_ind_set_by_random_vector_projection_kms_prim
                )
            elif find_ind_sets_strategy == 'multiple_random_vector_projection':
                self._partially_color_strategy = lambda g, L, colors: partially_color_graph_by_vector_projections(
                    g,
                    L,
                    colors,
                    find_ind_sets_strategy=find_multiple_ind_sets_by_random_vector_projections
                )
            elif find_ind_sets_strategy == 'clustering':
                self._partially_color_strategy = lambda g, L, colors: partially_color_graph_by_vector_projections(
                    g,
                    L,
                    colors,
                    find_ind_sets_strategy=find_ind_set_by_clustering
                )
            else:
                raise Exception('Unknown find_ind_set_strategy')
        else:
            raise Exception('Unknown partial coloring strategy')

        # Assign Wigderson strategy
        if wigderson_strategy == 'no_wigderson':
            self._wigderson_strategy = no_wigderson_strategy
        elif wigderson_strategy == 'recursive_wigderson':
            self._wigderson_strategy = recursive_wigderson_strategy
        else:
            raise Exception('Unknown Wigderson strategy')

        # Assign name
        if alg_name is not None:
            self.name = alg_name
        else:
            self.name = "pcs: " + partial_color_strategy + " ws: " + wigderson_strategy

    def color_graph(self, g, colors=None, verbose=True, precolors=None):
        """Colors graph using vector coloring algorithm.

        Args:
            g (Graph): Graph to be colored.
            colors (dict): Partial coloring of g
            verbose (bool): Set verbosity level e.g. for solver.
            precolors (dict): Full legal precoloring of graph g.

        Returns:
            dict: Global vertex-color dictionary indexed from 0 to g.number_of_nodes()-1.
        """

        if g.number_of_selfloops() > 0:
            raise Exception('Graph contains selfloops')

        if colors is None:
            colors = {v: -1 for v in g.nodes()}
        else:
            pass  # TODO: Delete colored vertices and edges; change color values to start from 0 continuously

        logging.info('Starting color_graph procedure on a graph with {0} vertices and {1} edges...'.format(
            g.number_of_nodes(), g.number_of_edges()))

        max_iterations = g.number_of_nodes() * 2  # is it a good boundary?
        current_g = g.copy()

        it = 0
        while (current_g.number_of_nodes() >= 0 and -1 in set(colors.values())) and it < max_iterations:
            it += 1
            logging.info('\n')
            logging.info('Starting iteration nr {0} of main loop...'.format(it))
            if current_g.number_of_nodes() > 1 and current_g.number_of_edges() > 0:
                L = compute_vector_coloring(current_g, precolors, strict=False, verbose=verbose, iteration=it)
                if it == 1:
                    if self._wigderson_strategy(current_g, colors, L):
                        continue  # Wigderson colored some vertices so we need to recompute vector coloring
                current_nodes = current_g.number_of_nodes()
                while current_g.number_of_nodes() == current_nodes:
                    self._partially_color_strategy(current_g, L, colors)
            elif current_g.number_of_nodes() == 1:
                colors[list(current_g.nodes())[0]] = get_lowest_legal_color(g, list(current_g.nodes())[0], colors)
                # colors[list(current_g.nodes())[0]] = max(colors.values()) + 1
                break
            elif current_g.number_of_edges() == 0:
                new_color = max(colors.values()) + 1
                for v in current_g.nodes():
                    colors[v] = new_color
                break
            else:
                break

        return colors

    def get_algorithm_name(self):
        return self.name


def compute_vector_coloring(g, precolors, strict=False, verbose=False, iteration=False):
    """Computes vector coloring on the basis of matrix coloring using Cholesky decomposition.

        Args:
            M (2-dim matrix): Matrix-coloring of the graph.

        Returns:
              2-dim matrix: Rows of this matrix are vectors of computed vector coloring.
        """

    def cholesky_factorize(m):
        """Returns L such that M = LL^T.

            According to https://en.wikipedia.org/wiki/Cholesky_decomposition#Proof_for_positive_semi-definite_matrices
                if L is positive semi-definite then we can turn it into positive definite by adding eps*I.

            We can also perform LDL' decomposition and set L = LD^(1/2) - it works in Matlab even though M is singular.

            It sometimes returns an error if M was computed with big tolerance for error.

            Args:
                m (2-dim matrix): Positive semidefinite matrix to be factorized.

            Returns:
                L (2-dim matrix): Cholesky factorization of M such that M = LL^T.
            """

        logging.info('Starting Cholesky factorization...')

        eps = 10e-8
        L = m

        for i in range(L.shape[0]):
            L[i, i] = L[i, i] + eps  # TODO: Should I normalize the output vector coloring?

        L = np.linalg.cholesky(L)

        # Probably don't need these lines
        for i in range(L.shape[0]):
            for j in range(i + 1, L.shape[0]):
                L[i, j] = 0

        logging.info('Cholesky factorization computed')
        return L

    def compute_matrix_coloring(g, precolors, strict=False, verbose=False):
        """Finds matrix coloring M of graph g using Mosek solver.

        Maybe we can add epsilon to SDP constraints instead of 'solve' parameters?

        For some reason optimal value of alpha is greater than value computed from M below if SDP is solved with big
            tolerance for error

        Args:
            g (Graph): Graph to be processed.
            strict (bool): Are we looking for strict vector coloring.
            verbose (bool): Sets verbosity level of solver.
            precolors (dict): Full legal coloring of graph G used to obtain good starting point for solver.

        Returns:
            2-dim matrix: Matrix coloring of graph G.
        """

        logging.info('Computing matrix coloring of graph with {0} nodes and {1} edges...'.format(
            g.number_of_nodes(), g.number_of_edges()
        ))

        dsatur_coloring = nx.algorithms.coloring.greedy_color(g, strategy='DSATUR')

        with Model() as M:

            # Variables
            n = g.number_of_nodes()
            alpha = M.variable("alpha", Domain.lessThan(0.))
            m = M.variable(Domain.inPSDCone(n))

            # Constraints
            M.constraint("C1", m.diag(), Domain.equalsTo(1.0))
            for i in range(n):
                for j in range(n):
                    if i > j and has_edge_between_ith_and_jth(g, i, j):
                        if strict:
                            M.constraint('C{0}-{1}'.format(i, j), Expr.sub(m.index(i, j), alpha),
                                         Domain.equalsTo(0.))
                        else:
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

    # if iteration == 1 and vector_coloring_in_file(g, strict):
    # l = read_vector_coloring_from_file(g, strict)
    # else:
    m = compute_matrix_coloring(g, precolors, strict=False, verbose=False)
    l = cholesky_factorize(m)
    if iteration == 1:
        save_vector_coloring_to_file(g, strict, l)

    return l


def compute_optimal_coloring_lp(g, colors=None, verbose=False):
    """Computes optimal coloring using linear programming."""

    with Model() as M:

        n = g.number_of_nodes()

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
                if i > j and has_edge_between_ith_and_jth(g, i, j):
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

        for i, v in enumerate(sorted(list(g.nodes()))):
            for c in range(n):
                if x.index(i, c).level() == 1.0:
                    coloring[v] = c
                    break

    return coloring


def compute_optimal_coloring_dp(g, colors=None, verbose=False):
    """Computes optimal coloring using dynamic programming."""

    T_Sets = {w: [] for r in range(g.number_of_nodes() + 1) for w in itertools.combinations(g.nodes(), r)}
    T = {w: -1 for r in range(g.number_of_nodes() + 1) for w in itertools.combinations(g.nodes(), r)}  # set(w)
    T[()] = 0

    for w in T.keys():
        if len(w) == 1:
            T[w] = 1
            T_Sets[w] = [w]

    for w in sorted(T.keys(), key=len):
        if len(w) <= 1:
            continue

        min_chi = g.number_of_nodes()
        min_s = []
        subsets = [s for r in range(1, len(w) + 1) for s in itertools.combinations(w, r)]  # non-empty subsets

        for s in subsets:
            if len(s) == len(w):
                g_s = g.subgraph(s)
                if g_s.number_of_edges() > 0:
                    continue
            else:
                if T[s] > 1:  # S is not independent
                    continue

            temp = list(set(w) - set(s))
            temp.sort()
            temp = tuple(temp)

            if T[temp] < min_chi:
                min_chi = T[temp]
                min_s = s

        T[w] = min_chi + 1
        T_Sets[w] = min_s

    # Now compute the coloring
    ind_sets = []
    vertices = tuple(g.nodes())
    while vertices:
        i_set = T_Sets[vertices]
        vertices = tuple(v for v in vertices if v not in i_set)
        ind_sets.append(i_set)

    coloring = {}
    clr = -1
    for v_set in ind_sets:
        clr += 1
        for v in v_set:
            coloring[v] = clr

    return coloring
