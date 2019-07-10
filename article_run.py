import math

import seaborn as sns

from graph_io import read_graph_from_file
from solver import *


def compute_directed_vc(graph, beta_factors, sdp_type):
    directed_vc, directed_mc = compute_directed_vector_coloring(graph, beta_factors, sdp_type, verbose=True,
                                                                is_alpha_constrained=False)
    return directed_vc, directed_mc


def compute_standard_vc(graph, sdp_type):
    vc = compute_vector_coloring(graph, sdp_type, True)
    return vc


def better_ind_sets(graph, ind_sets1, ind_sets2):
    """Returns true if the first list of independent sets is better than the second."""

    if ind_sets2 is None or len(ind_sets2) == 0:
        return True

    if ind_sets1 is None or len(ind_sets1) == 0:
        return False

    temp_colors1 = {v: -1 for s in ind_sets1 for v in s}
    temp_colors2 = {v: -1 for s in ind_sets2 for v in s}

    if len(set(temp_colors2.values())) == 0:
        return False

    if len(set(temp_colors1.values())) == 0:
        return False

    color = 0
    for ind_set in ind_sets1:
        color += 1
        for v in ind_set:
            temp_colors1[v] = color

    color = 0
    for ind_set in ind_sets2:
        color += 1
        for v in ind_set:
            temp_colors2[v] = color

    avg1 = len(temp_colors1.keys()) / float(len(set(temp_colors1.values())))
    avg2 = len(temp_colors2.keys()) / float(len(set(temp_colors2.values())))

    return avg1 > avg2


def find_ind_sets_by_directed_vector_projection(graph, L, init_params, color_by_independent_sets_params, nr_of_sets=1):

    def compute_c_opt(graph, L):
        c = 0.4  # TODO: mozna to wyznaczyc, aby miec pewnosc (jak z klastrami)

        return c

    c_opt = compute_c_opt(graph, L)

    n = len(L)
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(graph.nodes()))}

    best_ind_sets = []
    best_extracted_ratios = []
    it = 0
    last_change = 0
    while it < color_by_independent_sets_params['nr_of_random_vectors_tried'] \
            and it - last_change < color_by_independent_sets_params['max_nr_of_random_vectors_without_change']:
        it += 1

        ind_sets = []
        extracted_ratios = []
        for i in range(nr_of_sets):
            r = -L[n - 1]
            x = np.dot(L,
                       r)  # tego nie trzeba liczyc, bo to po prostu -ostatni wiersz z M (kolorowania macierzoweg); ta wiedza pozwoli moze odpowiednio dobrac c
            best_ind_set = []
            best_extracted_ratio = 0
            for c in np.linspace(
                    c_opt * color_by_independent_sets_params['c_param_lower_factor'],
                    c_opt * color_by_independent_sets_params['c_param_upper_factor'],
                    num=color_by_independent_sets_params['nr_of_c_params_tried_per_random_vector']):
                current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c}
                current_subgraph_edges = {(i, j) for i, j in graph.edges() if
                                          (i in current_subgraph_nodes and j in current_subgraph_nodes)}

                ind_set, extracted_ratio = extract_independent_subset(
                    current_subgraph_nodes, current_subgraph_edges,
                    strategy=init_params['independent_set_extraction_strategy'],
                    verbose=True)
                ind_set = [ind_set]

                if better_ind_sets(graph, ind_set, best_ind_set):
                    best_ind_set = ind_set
                    best_extracted_ratio = extracted_ratio
                    last_change = it

            ind_sets.extend(best_ind_set)
            extracted_ratios.append(best_extracted_ratio)

        if better_ind_sets(graph, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets
            best_extracted_ratios = extracted_ratio

    return best_ind_sets, best_extracted_ratios


def find_ind_sets_by_standard_vector_projection(graph, L, init_params, color_by_independent_sets_params, nr_of_sets=1):
    """KMS according to Arora, Chlamtac, Charikar.

    Tries to return nr_of_sets but might return less.
    """

    def compute_c_opt(graph, L):
        """Computes optimal c parameter according to KMS.

        Args:
            graph (nx.Graph): Graph
            L (2-dim array): Vector coloring of graph
        """

        max_degree = max(dict(graph.degree()).values())
        k = find_number_of_vector_colors_from_vector_coloring(graph, L)
        temp = (2 * (k - 2) * math.log(max_degree)) / k
        if temp >= 0:
            c = math.sqrt(temp)
        else:
            c = 0.0

        return c

    c_opt = compute_c_opt(graph, L)

    n = graph.number_of_nodes()
    inv_vertices_mapping = {i: v for i, v in enumerate(sorted(graph.nodes()))}

    best_ind_sets = []
    it = 0
    last_change = 0
    while it < color_by_independent_sets_params['nr_of_random_vectors_tried'] \
            and it - last_change < color_by_independent_sets_params['max_nr_of_random_vectors_without_change']:
        it += 1

        ind_sets = []
        for i in range(nr_of_sets):
            r = np.random.normal(0, 1, n)
            x = np.dot(L, r)
            best_ind_set = []
            for c in np.linspace(
                    c_opt * color_by_independent_sets_params['c_param_lower_factor'],
                    c_opt * color_by_independent_sets_params['c_param_upper_factor'],
                    num=color_by_independent_sets_params['nr_of_c_params_tried_per_random_vector']):
                current_subgraph_nodes = {inv_vertices_mapping[i] for i, v in enumerate(x) if v >= c}
                current_subgraph_edges = {(i, j) for i, j in graph.edges() if
                                          (i in current_subgraph_nodes and j in current_subgraph_nodes)}

                ind_set = [extract_independent_subset(
                    current_subgraph_nodes, current_subgraph_edges,
                    strategy=init_params['independent_set_extraction_strategy'],
                    verbose=True)]

                if better_ind_sets(graph, ind_set, best_ind_set):
                    best_ind_set = ind_set
                    last_change = it

            ind_sets.extend(best_ind_set)

        if better_ind_sets(graph, ind_sets, best_ind_sets):
            best_ind_sets = ind_sets

    return best_ind_sets


def draw_distributions(Ms, betas):
    for M, beta in zip(Ms, betas):
        draw_dummy_dot_products(M, beta)

    # function to show the plot
    plt.legend()
    plt.show()


def draw_dummy_dot_products(M, beta_factor):
    """M - matrix coloring. Last row (or column) consists of dummy vector dot products with others"""

    n = M.shape[0]
    dummy_dot_products = M[n - 1]

    sns.set(color_codes=True)
    sns.distplot(dummy_dot_products[0:n - 1], label='{0}'.format(beta_factor))

    # # x axis values
    # x = [i for i in range(n-1)]
    # # corresponding y axis values
    # y = dummy_dot_products[0:n-1]
    #
    # # plotting the points
    # plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
    #          marker='o', markerfacecolor='blue', markersize=12)
    #
    # # setting x and y axis range
    # # plt.ylim(1, 8)
    # # plt.xlim(1, 8)
    #
    # # naming the x axis
    # plt.xlabel('x - axis')
    # # naming the y axis
    # plt.ylabel('y - axis')
    #
    # # giving a title to my graph
    # plt.title('Some cool customizations!')
    #


def compute_directed_vector_coloring(graph, beta_factors, sdp_type, verbose, is_alpha_constrained):
    M = compute_directed_matrix_coloring(graph, beta_factors, sdp_type, verbose, is_alpha_constrained)
    L = cholesky_factorize(M)  # TODO: obejrzyj sobie M dla roznych grafow roznych c

    return L, M


# graph_name = "dimacs/DSJC125.1"
# graph = create_star(250)

# graph_name = "other/grotzsch"
graph_name = "dimacs/DSJC125.1"
graph = read_graph_from_file(folder_name="", graph_name=graph_name, graph_type=None, starting_index=0)

standard_vc = compute_standard_vc(graph, 'nonstrict')

# for beta_factor in np.linspace(0.1, 2.0, 10):
beta_factors = [1.0] * graph.number_of_nodes()  # czy beta beda ujemne? jesli tak, to factors inny znak
directed_vc, directed_mc = compute_directed_vc(graph, beta_factors, 'nonstrict')

draw_distributions([directed_mc], [beta_factors[0]])


# wydobadz wiele razy jeden zbior niezalezny i policz srednia - dla random vector projection i directed vector projection

init_params = {
    'partial_color_strategy': 'color_by_independent_sets',
    'sdp_type': 'nonstrict',
    'independent_set_extraction_strategy': 'max_degree_first',
    'alg_name': '7-5-1830-algo',
    'find_independent_sets_strategy': 'random_vector_projection',
    'deterministic': True,
}

standard_vc_params = {
    'nr_of_times_restarting_ind_set_strategy': 1,  # it enables parallelism so at least 8
    'nr_of_random_vectors_tried': 1,
    'max_nr_of_random_vectors_without_change': 100,
    'c_param_lower_factor': 1.0,
    'c_param_upper_factor': 1.0,
    'nr_of_c_params_tried_per_random_vector': 1,
    'nr_of_cluster_sizes_to_check': 1,
    'cluster_size_lower_factor': 1.0,  # Makes no sense to set it much lower than 1.0
    'cluster_size_upper_factor': 1.0,
}

directed_vc_params = standard_vc_params

directed_ind_sets = []
standard_ind_sets = []
directed_ind_sets_sizes = []
standard_ind_sets_sizes = []
directed_extracted_ratios = []

samples = 100
for i in range(samples):
    print 'Directed: '
    found_directed_ind_sets, extracted_ratio = find_ind_sets_by_directed_vector_projection(graph, directed_vc,
                                                                                           init_params,
                                                                          directed_vc_params, 1)
    directed_ind_sets_sizes.append(len(found_directed_ind_sets[0]))
    directed_extracted_ratios.append(extracted_ratio)

    print 'Standard: '
    found_standard_ind_sets = find_ind_sets_by_standard_vector_projection(graph, standard_vc, init_params,
                                                                          standard_vc_params, 1)
    standard_ind_sets_sizes.append(len(found_standard_ind_sets[0]))

avg_directed_ind_set_size = float(sum(directed_ind_sets_sizes)) / float(samples)
avg_standard_ind_set_size = float(sum(standard_ind_sets_sizes)) / float(samples)
avg_directed_extracted_ratio = float(sum(directed_extracted_ratios)) / float(samples)

print 'avg_directed_extracted_ratios: {0}'.format(avg_directed_extracted_ratio)
print 'avg_directed_ind_set_size: {0}'.format(avg_directed_ind_set_size)
print 'avg_standard_ind_set_size: {0}'.format(avg_standard_ind_set_size)

# 2. sprawdz jak taka ekstrakcja wplywa na kolorowanie

# TODO: jak w zaleznosci od c zmienia sie ind_set_size oraz extracted_ratio (wykres)
