import logging
import os
import sys

import cvxpy
from mosek.fusion import *
from numpy.linalg import LinAlgError

from coloring.algorithm_helper import *
from configuration import algorithm_options_config, paths_config

mosek_params_default = {
    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-7,
    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
    'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-8,
    'MSK_DPAR_INTPNT_CO_TOL_INFEAS': 1e-10,
    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
    'MSK_DPAR_SEMIDEFINITE_TOL_APPROX': 1e-10
}

mosek_params = {
    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-4,
    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-5,
    'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-5,
    'MSK_DPAR_INTPNT_CO_TOL_INFEAS': 1e-7,
    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-5,
    'MSK_DPAR_SEMIDEFINITE_TOL_APPROX': 1e-10
}


def compute_dummy_vector_coloring(graph, beta_factor_strategy, alpha_upper_bound):
    """Compute dummy vector coloring.

    :param graph: (nx.Graph)
    :param beta_factor_strategy: strategy describing how to choose beta factors
    :param alpha_upper_bound: upper bound for alpha parameter
    :return: (n+1 x n+1 matrix) dummy vector coloring (rows are vectors of the coloring)
    """

    dummy_matrix_coloring = compute_dummy_matrix_coloring(graph, beta_factor_strategy, alpha_upper_bound)
    dummy_vector_coloring = cholesky_factorize(dummy_matrix_coloring)

    # draw_distributions(dummy_matrix_coloring, 1.0)
    return dummy_vector_coloring


def compute_vector_coloring(graph, sdp_type):
    """Computes sdp_type (not dummy) vector coloring of graph using Cholesky decomposition.

    :param graph: (nx.Graph)
    :param sdp_type:  Non-strict, Strict or Strong coloring.
    :return: (n x n matrix) Rows of this matrix are vectors of computed vector coloring.
    """

    matrix_coloring = compute_matrix_coloring(graph, sdp_type)
    vector_coloring = cholesky_factorize(matrix_coloring)

    return vector_coloring


def compute_dummy_matrix_coloring(graph, beta_factor_strategy, alpha_upper_bound):
    """Compute dummy matrix coloring.

    :param graph: (nx.Graph)
    :param beta_factor_strategy: strategy describing how to choose beta factors
    :param alpha_upper_bound: upper bound for alpha parameter
    :return: ((n+1) x (n+1) matrix) dummy matrix coloring, with n+1-st vector being dummy vector (not belonging to the graph)

    """
    if algorithm_options_config.solver_name == 'mosek':
        dummy_matrix_coloring, alpha_opt = find_dummy_matrix_coloring_mosek(
            graph, beta_factor_strategy, alpha_upper_bound)
    else:
        raise Exception('Unknown solver name')

    logging.debug('Found matrix {0}-coloring'.format(1 - 1 / alpha_opt))

    return dummy_matrix_coloring


def compute_matrix_coloring(graph, sdp_type):
    """Finds matrix coloring (not dummy) of graph using sdp solver.

    Notes:
    Maybe we can add epsilon to SDP constraints instead of 'solve' parameters?
    For some reason optimal value of alpha is greater than value computed from M below if SDP is solved with big
    tolerance for error

    :param graph: (nx.Graph) Graph to be processed.
    :param sdp_type: Non-strict, Strict or Strong vector coloring.
    :return: (n x n matrix): Matrix coloring of graph.
    """

    if algorithm_options_config.solver_name == 'mosek':
        matrix_coloring, alpha_opt = find_standard_matrix_coloring_mosek(graph, sdp_type)
    elif algorithm_options_config.solver_name == 'cvxopt':
        matrix_coloring, alpha_opt = solve_cvxopt(graph, sdp_type)
    else:
        raise Exception('Unknown solver name')

    logging.debug('Found matrix {0}-coloring'.format(1 - 1 / alpha_opt))

    return matrix_coloring


def cholesky_factorize(m):
    """Returns L such that m = LL^T.

    According to https://en.wikipedia.org/wiki/Cholesky_decomposition#Proof_for_positive_semi-definite_matrices
    if L is positive semi-definite then we can turn it into positive definite by adding eps*I.

    We can also perform LDL' decomposition and set L = LD^(1/2) - it works in Matlab even though M is singular.

    It sometimes returns an error if M was computed with big tolerance for error.

    :param m: (2-dim matrix) Positive semidefinite matrix to be factorized.
    :return L: (2-dim matrix) Cholesky factorization of M such that m = LL^T.
    """

    try:
        m = np.linalg.cholesky(m)
    except LinAlgError:
        # it is a hack that slightly changes the matrix so that it becomes positive semi-definite
        eps = 1e-7
        for i in range(m.shape[0]):
            m[i, i] = m[i, i] + eps
        m = np.linalg.cholesky(m)

    return m


def find_dummy_matrix_coloring_mosek(graph, beta_factors_strategy, alpha_upper_bound):
    """Finds dummy matrix coloring using Mosek solver.

    Minimized function is:
        n*alpha + beta_factor1*beta1 + beta_factor2*beta2 + ... + beta_factorN*betaN,
    where the constant beta_factors determine how strong is mutual repulsion of connected vector-vertices vs. how strong
    is the repulsion of a given vector-vertex from dummy vector.

    n - number of vertices in graph
    alpha - variable upper bound of dot-product between non-dummy vertex-vectors
    betaK - variable upper bound of dot-product between k-th vertex-vector and dummy vertex-vector
    beta_factorK - constant that determines how important is the repulsion form dummy vs. repulsion from neighboring vertex-vectors

    :param graph: (nx.Graph)
    :param beta_factors_strategy: strategy describing how to choose beta factors
    :param alpha_upper_bound: upper bound for alpha parameter
    :return: ((n+1) x (n+1) matrix) dummy matrix coloring and value of 'alpha' that was achieved in optimum
    """

    with Model() as Mdl:
        # Variables
        n = graph.number_of_nodes()
        alpha = Mdl.variable(Domain.inRange(-1, alpha_upper_bound))
        m = Mdl.variable(Domain.inPSDCone(n + 1))
        betas = Mdl.variable(n, Domain.inRange(-1, 1))

        # Constraints
        add_mosek_dummy_constraints(graph, Mdl, m, alpha, betas)

        # Objective
        beta_factors = create_beta_factors(graph, beta_factors_strategy)
        Mdl.objective(ObjectiveSense.Minimize, Expr.add(Expr.mul(n, alpha), Expr.sum(Expr.mulElm(beta_factors, betas))))

        if algorithm_options_config.solver_verbose:
            if algorithm_options_config.solver_output == 'file':
                with open(os.path.join(paths_config.logs_directory(), "logs"), 'w') as outfile:
                    Mdl.setLogHandler(outfile)
            else:
                Mdl.setLogHandler(sys.stdout)

        Mdl.solve()

        alpha_opt = alpha.level()[0]
        level = m.level()
        # TODO: check if below is correct and maybe simplify it so that it makes sense
        # TODO: possibly also return values of beta, not only alfa, as it may be useful to log it
        dummy_matrix_coloring = [[level[j * (n + 1) + i] for i in range(n + 1)] for j in range(n + 1)]
        dummy_matrix_coloring = np.array(dummy_matrix_coloring)

        return dummy_matrix_coloring, alpha_opt


def find_standard_matrix_coloring_mosek(graph, sdp_type):
    """Finds standard matrix coloring using Mosek solver.

    Minimized function is:
        alpha
    where:

    n - number of vertices in graph
    alpha - variable upper bound of dot-product between vertex-vectors

    :param graph: (nx.Graph)
    :param sdp_type:  Non-strict, Strict or Strong vector coloring.
    :return: (n x n matrix) standard matrix coloring
    """

    with Model() as Mdl:
        # Variables
        n = graph.number_of_nodes()
        alpha = Mdl.variable(Domain.lessThan(0.))
        m = Mdl.variable(Domain.inPSDCone(n))

        if n <= algorithm_options_config.sdp_strong_threshold:
            sdp_type = 'strong'

        # Constraints
        if sdp_type == 'nonstrict':
            add_mosek_sdp_nonstrict_constraints(graph, Mdl, m, alpha)
        elif sdp_type == 'strict':
            add_mosek_sdp_strict_constraints(graph, Mdl, m, alpha)
        elif sdp_type == 'strong':
            add_mosek_sdp_strong_constraints(graph, Mdl, m, alpha)

        # Objective
        Mdl.objective(ObjectiveSense.Minimize, alpha)

        if algorithm_options_config.solver_verbose:
            if algorithm_options_config.solver_output == 'file':
                with open(os.path.join(paths_config.logs_directory(), "logs"), 'w') as outfile:
                    Mdl.setLogHandler(outfile)
            else:
                Mdl.setLogHandler(sys.stdout)

        Mdl.solve()

        alpha_opt = alpha.level()[0]
        level = m.level()
        result = [[level[j * n + i] for i in range(n)] for j in range(n)]
        result = np.array(result)

        return result, alpha_opt


def create_beta_factors(graph, beta_factors_strategy):
    """Creates n-element list of beta factors that define how strongly vertex-vectors are pushed from dummy vertex-vector
        while solving matrix coloring sdp optimization problem.

    :param graph: (nx.Graph)
    :param beta_factors_strategy: strategy of choosing beta factors
    :return: n-element list of beta factors - one for each real vertex-vector
    """

    n = graph.number_of_nodes()

    if beta_factors_strategy['name'] == "uniform":
        beta_factors = [beta_factors_strategy['factor']] * n
    else:
        raise Exception("Unknown beta factors strategy")

    return beta_factors


def add_mosek_dummy_constraints(graph, model, matrix, alpha, betas):
    """Add constraints to mosek solver model for dummy matrix coloring.

    :param graph: (nx.Graph)
    :param model: mosek solver model
    :param matrix: matrix of mosek solver model representing dot products of vector-vertices
    :param alpha: alpha factor that is an upper bound on dot product between non-dummy vertex-vectors
    :param betas: n-element list of beta factors that define how strongly vertex-vectors are pushed from dummy vertex-vector
        while solving matrix coloring sdp optimization problem
    """

    n = graph.number_of_nodes()

    model.constraint(matrix.diag(), Domain.equalsTo(1.0))
    for i in range(n):
        for j in range(i):
            if has_edge_between_ith_and_jth(graph, i, j):
                model.constraint(Expr.sub(matrix.index(i, j), alpha), Domain.lessThan(0.))

    for i in range(n):
        model.constraint(Expr.sub(matrix.index(n, i), betas.index(i)), Domain.lessThan(0.))


def add_mosek_sdp_nonstrict_constraints(graph, model, matrix, alpha):
    """Add constraints to mosek solver model for nonstrict standard matrix coloring.

    :param graph: (nx.Graph)
    :param model: mosek solver model
    :param matrix: matrix of mosek solver model representing dot products of vector-vertices
    :param alpha: alpha factor that is an upper bound on dot product between non-dummy vertex-vectors
    """

    n = graph.number_of_nodes()

    model.constraint(matrix.diag(), Domain.equalsTo(1.0))
    for i in range(n):
        for j in range(i):
            if has_edge_between_ith_and_jth(graph, i, j):
                model.constraint(Expr.sub(matrix.index(i, j), alpha), Domain.lessThan(0.))


def add_mosek_sdp_strict_constraints(graph, model, matrix, alpha):
    """Add constraints to mosek solver model for strict standard matrix coloring.

    :param graph: (nx.Graph)
    :param model: mosek solver model
    :param matrix: matrix of mosek solver model representing dot products of vector-vertices
    :param alpha: alpha factor that is an upper bound on dot product between non-dummy vertex-vectors
    """

    n = graph.number_of_nodes()

    model.constraint(matrix.diag(), Domain.equalsTo(1.0))
    for i in range(n):
        for j in range(i):
            if has_edge_between_ith_and_jth(graph, i, j):
                model.constraint(Expr.sub(matrix.index(i, j), alpha), Domain.equalsTo(0.))


def add_mosek_sdp_strong_constraints(graph, model, matrix, alpha):
    """Add constraints to mosek solver model for strong standard matrix coloring.

    :param graph: (nx.Graph)
    :param model: mosek solver model
    :param matrix: matrix of mosek solver model representing dot products of vector-vertices
    :param alpha: alpha factor that is an upper bound on dot product between non-dummy vertex-vectors
    """

    n = graph.number_of_nodes()

    model.constraint(matrix.diag(), Domain.equalsTo(1.0))
    for i in range(n):
        for j in range(i):
            if has_edge_between_ith_and_jth(graph, i, j):
                model.constraint(Expr.sub(matrix.index(i, j), alpha), Domain.equalsTo(0.))
            else:
                model.constraint(Expr.add(matrix.index(i, j), alpha), Domain.greaterThan(0.))


def solve_cvxopt(graph, sdp_type):
    """Finds matrix coloring of graph using cvxopt solver.

    I must be doing something wrong with the model definition - too many constraints and variables

    :param graph: (nx.Graph)
    :param sdp_type: unused
    :return: resulting matrix coloring and optimal value of alpha parameter
    """

    n = graph.number_of_nodes()

    # Variables
    alpha = cvxpy.Variable()
    mat = cvxpy.Variable((n, n), PSD=True)

    # Constraints (can be done using trace as well)
    constraints = []
    for i in range(n):
        constraints += [mat[i, i] == 1]

    for i in range(n):
        for j in range(n):
            if i > j and has_edge_between_ith_and_jth(graph, i, j):
                constraints += [mat[i, j] <= alpha]

    # Objective
    objective = cvxpy.Minimize(alpha)

    # Create problem instance
    problem = cvxpy.Problem(objective, constraints)

    # Solve
    try:
        problem.solve(
            solver=cvxpy.MOSEK,
            verbose=algorithm_options_config.solver_verbose,
            warm_start=True,
            mosek_params=mosek_params)
        alpha_opt = alpha.value
        result = mat.value
    except cvxpy.error.SolverError:
        print '\nError in mosek, changing to cvxopt\n'
        problem.solve(
            solver=cvxpy.CVXOPT,
            verbose=algorithm_options_config.solver_verbose,
            warm_start=True)
        alpha_opt = alpha.value
        result = mat.value

    return result, alpha_opt
