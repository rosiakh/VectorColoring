from result_io import *
from result_processing import *

no_result_char = "--"


def create_and_save_latex_document(result_seed=None):
    """Create results in form of latex source document and save it to file (it should compile on overleaf.com)

    Created latex source is saved in "paths_config.latex_result_path"

    :param result_seed: result seed associated with results that we want to present in latex
    """

    latex_document = ""
    with open(paths_config.latex_document_top_path) as f:
        for line in f:
            latex_document += line

    latex_document += create_latex_tables(result_seed)
    latex_document += "\n\\end{document}"

    with open(paths_config.latex_result_path, 'w') as outfile:
        outfile.write(latex_document)


def create_and_save_latex_tables(result_seed=None):
    """Create results in form of latex source and save it to file (which won't compile - it requires some boilerplate latex)

    Created (non-compiling) latex source is saved in "paths_config.latex_result_path"

    :param result_seed: result seed associated with results that we want to present in latex
    """

    latex_text = create_latex_tables(result_seed)
    with open(paths_config.latex_result_path, 'w') as outfile:
        outfile.write(latex_text)


def create_latex_tables(result_seed=None):
    """Create results in form of latex source and save it to file (which won't compile - it requires some boilerplate latex)

        Created (non-compiling) latex source is saved in "paths_config.latex_result_path"

        :param result_seed: result seed associated with results that we want to present in latex
        :return (string) latex text
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

        return latex_text


def create_latex_table_string(results_directory, table_caption, data_to_save_property, is_float_property, is_bold_min,
                              algorithm_names):
    """Creates latex code representing one result table.

    :param results_directory: directory containing results that are to be put in the table
    :param table_caption: table caption
    :param data_to_save_property: what property of the result data is to be put in the table (e.g. "min_nr_of_colors"
        or "avg_time")
    :param is_float_property: is the property of type float
    :param is_bold_min: if true, then the minimal value of the row will be made bold; if false, the maximal value will
        made bold
    :param algorithm_names: list of names of algorithms that were used to create the result data
    :return: latex code representing a table of results
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
    """Creates description of the results by presenting used algorithms.

    :param algorithm_names: names of algorithms that are to be put in the legend
    :return: description of algorithms in form of latex code
    """

    latex_text = "\\begin{itemize}\n"
    for i, algorithm_name in enumerate(algorithm_names):
        latex_text += \
            "\\item " + get_algorithm_letter(algorithm_name) + str(i) + " = " + algorithm_name.replace("_", "\_") + "\n"
    latex_text += "\\end{itemize}\n"

    return latex_text


def get_algorithm_names(results_directory):
    """Gets names of all algorithms used to obtain the results in results_directory based on algorithm info filename.

    Algorithm info filename is "results_directory/paths_config.algorithm_info_filename"

    :param results_directory:
    :return: list of algorithm names
    """

    algorithm_names = []
    with open(join(results_directory, paths_config.algorithm_info_filename), 'r') as algorithm_info_file:
        for algorithm_name in algorithm_info_file:
            algorithm_names.append(algorithm_name.strip('\n'))

    return algorithm_names


def create_latex_table_row(graph_name, sorted_algorithm_names, results, data_to_save_property, is_float_property,
                           is_bold_min, is_round_float=True):
    """Creates latex code representing one row of a table

    :param graph_name: unused
    :param sorted_algorithm_names: names of algorithms for which results are presented in the row
    :param results: results that will be put in the row
    :param data_to_save_property: name of the property that will be read from results and put into the row (e.g.
        "min_nr_of_colors" or "avg_time")
    :param is_float_property: is the property of type float
    :param is_bold_min: if true, then the minimal value of the row will be made bold; if false, the maximal value will
        made bold
    :param is_round_float: should the property value be rounded if it is of type float
    :return: latex code representing one row of a table
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
        Graph & Vertices & Density & algName1 & algName2 & .... & algNameN
        \hline

    :param sorted_algorithm_names: names of algorithms to be put as column headers from left to right (starting with
        4th column)
    :param caption: table caption
    :return: latex code representing top part of the table with header and caption
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
    """Make extremal (minimal or maximal) values in given latex row bold by using \textbf{...}

    :param latex_row: latex code representing a row in a table
    :param from_column: first column that we are taking into account (numbering from left)
    :param to_column: last column that we are taking into account (numbering from left)
    :param is_bold_min: if true, then the minimal value of the row will be made bold; if false, the maximal value will
        made bold
    :return: latex code with row with extremal value made bold
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
