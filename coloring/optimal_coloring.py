""" Module containing optimal coloring algorithms. """

import itertools
import sys

from mosek.fusion import *

from algorithm_helper import *
from configuration.algorithm_options_config import *


def compute_optimal_coloring_lp(graph):
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

        if general_verbosity:
            M.setLogHandler(sys.stdout)

        M.solve()

        coloring = {}

        for i, v in enumerate(sorted(list(graph.nodes()))):
            for c in range(n):
                if x.index(i, c).level() == 1.0:
                    coloring[v] = c
                    break

    return coloring


def compute_optimal_coloring_dp(graph):
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
