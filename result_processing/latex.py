from result_io import *
from result_processing import *

no_result_char = "--"


def create_and_save_latex_tables(result_seed=None):
    """

    :param result_seed:
    :return:
    """

    if result_seed is None:
        result_seed = find_newest_result_seed()

    all_results_subdirs = all_subdirs_of(paths_config.results_directory(result_seed))
    algorithm_names = get_algorithm_names(paths_config.results_directory(result_seed))

    latex_text = ""
    latex_text += create_legend(algorithm_names)

    for results_subdir in all_results_subdirs:
        latex_text += create_latex_table_string(
            results_directory=results_subdir,
            table_caption=os.path.basename(os.path.normpath(results_subdir)) + " - Number of colors used",
            data_to_save_property="min_nr_of_colors",
            is_float_property=False,
            is_bold_min=True,
            algorithm_names=algorithm_names)
        latex_text = latex_text.replace("\_0 ", " ")
        latex_text += "\\begin{landscape}\\n"
        latex_text += create_latex_table_string(
            results_directory=results_subdir,
            table_caption=os.path.basename(os.path.normpath(results_subdir)) + " - Time of computation [s]",
            data_to_save_property="avg_time",
            is_float_property=True,
            is_bold_min=True,
            algorithm_names=algorithm_names)
        latex_text += "\\end{landscape}\n"

        with open(paths_config.latex_result_path, 'w') as outfile:
            outfile.write(latex_text)


def create_latex_table_string(results_directory, table_caption, data_to_save_property, is_float_property, is_bold_min,
                              algorithm_names):
    """

    :param results_directory:
    :param table_caption:
    :param data_to_save_property:
    :param is_float_property:
    :param is_bold_min:
    :param algorithm_names:
    :return:
    """

    sorted_graph_names = get_sorted_graph_names(results_directory)

    latex_text = ""

    latex_text += create_latex_table_top(algorithm_names, table_caption)
    for graph_name in sorted_graph_names:
        graph_results = load_algorithm_run_data_from_graph_directory(join(results_directory, graph_name))
        latex_text += create_latex_table_row(
            graph_name, algorithm_names, graph_results, data_to_save_property, is_float_property, is_bold_min)
    latex_text += create_latex_table_bottom()
    latex_text = latex_text.replace("_", "\\_")

    return latex_text


def create_legend(algorithm_names):
    """

    :param algorithm_names:
    :return:
    """

    latex_text = "\\begin{itemize}\n"
    for i, algorithm_name in enumerate(algorithm_names):
        latex_text += "\\item " + get_algorithm_letter(algorithm_name) + str(i) + " = " + algorithm_name + "\n"
    latex_text += "\\end{itemize}\n"

    return latex_text


def get_algorithm_names(results_directory):
    """

    :param results_directory:
    :return:
    """

    algorithm_names = []
    with open(join(results_directory, paths_config.algorithm_info_filename), 'r') as algorithm_info_file:
        for algorithm_name in algorithm_info_file:
            algorithm_names.append(algorithm_name.strip('\n'))

    return algorithm_names


def create_latex_table_row(graph_name, sorted_algorithm_names, results, data_to_save_property, is_float_property,
                           is_bold_min, is_round_float=True):
    """

    :param graph_name: unused
    :param sorted_algorithm_names:
    :param results:
    :param data_to_save_property:
    :param is_float_property:
    :param is_bold_min:
    :param is_round_float:
    :return:
    """

    latex_row = "{0} & {1} & {2:.2f}".format(
        results[results.keys()[0]][0]['graph_family'],
        results[results.keys()[0]][0]['graph_nr_of_vertices'],
        results[results.keys()[0]][0]['graph_density']
    )

    for algorithm_name in sorted_algorithm_names:
        if algorithm_name in results.keys() and len(results[algorithm_name]) > 0:
            algorithm_data_to_save = results[algorithm_name][0]
            if is_float_property:
                if is_round_float:
                    latex_row += " & {0}".format(int(round(algorithm_data_to_save[data_to_save_property])))
                else:
                    latex_row += " & {0:.2f}".format(algorithm_data_to_save[data_to_save_property])
            else:
                latex_row += " & {0}".format(algorithm_data_to_save[data_to_save_property])
        else:
            latex_row += " & {0}".format(no_result_char)

    latex_row = bold_extreme_value_latex_row(latex_row, 3, 3 + len(sorted_algorithm_names), is_bold_min)

    latex_row += " \\\\\n"
    return latex_row


def create_latex_table_top(sorted_algorithm_names, caption):
    """Create latex table top part of latex string
        \begin{longtable}[c]{|l|c|c|...|c|}
        \caption{some caption}
        \hline
        #TODO finish this docstring

    :param sorted_algorithm_names:
    :param caption:
    :return: latex table top part string
    """

    longtable = "\\begin{longtable}[c]{|l|c|c|" + "c|" * len(sorted_algorithm_names) + "}\n"
    caption = "\\caption{" + caption + "}\\\\\n\\hline\n"
    header = "Graph & Vertices & Density"
    for i, algorithm in enumerate(sorted_algorithm_names):
        header += " & " + get_algorithm_letter(algorithm) + str(i)
    header += "\\\\\n\\hline\n"

    return longtable + caption + header


def get_algorithm_letter(algorithm_name):
    """Get one letter abbreviation of coloring algorithm type based on its name
        D - dummy vector coloring
        R - random vector coloring
        G - greedy coloring
        A - all other

    :param algorithm_name: name of an algorithm
    :return: one letter abbreviation of algorithm type
    """

    alg_letter = "A"
    if "dummy" in algorithm_name:
        alg_letter = "D"
    elif "random" in algorithm_name:
        alg_letter = "R"
    elif "Greedy" in algorithm_name:
        alg_letter = "G"
    return alg_letter


def create_latex_table_bottom():
    """Create
        \end{longtable}
        part of latex table string

    :return: bottom of latex table
    """

    return "\\end{longtable}\n"


def bold_extreme_value_latex_row(latex_row, from_column, to_column, is_bold_min):
    """

    :param latex_row:
    :param from_column:
    :param to_column:
    :param is_bold_min:
    :return:
    """

    latex_row_values_with_special_chars = latex_row.split(' & ')[from_column:to_column]
    latex_row_values = [v for v in latex_row_values_with_special_chars if v != no_result_char]
    latex_row_values = map(lambda s: float(s), latex_row_values)

    if is_bold_min:
        extreme_value = min(latex_row_values)
    else:
        extreme_value = max(latex_row_values)

    bold_latex_row = latex_row.split(' & ')
    for i, value in enumerate(bold_latex_row[from_column:to_column]):
        new_value = value
        try:
            if float(value) == extreme_value:
                new_value = "\\textbf{" + value + "}"
        except ValueError:
            pass
        bold_latex_row[from_column + i] = new_value

    return ' & '.join(bold_latex_row)
