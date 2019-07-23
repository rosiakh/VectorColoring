from result_io import *
from result_processing import *


def create_and_save_latex_tables(result_seed=None):
    if result_seed is None:
        result_seed = find_newest_result_seed()

    all_subdirs = all_subdirs_of(paths_config.results_directory(result_seed))

    latex_text = ""
    latex_text += create_legend(all_subdirs[len(all_subdirs) - 1])

    for subdir in all_subdirs:
        latex_text += create_latex_table_string(
            result_directory=subdir,
            table_caption=os.path.basename(os.path.normpath(subdir)) + " - Number of colors used",
            data_to_save_property="min_nr_of_colors",
            is_float_property=False,
            is_bold_min=True)
        latex_text += create_latex_table_string(
            result_directory=subdir,
            table_caption=os.path.basename(os.path.normpath(subdir)) + " - Time of computation",
            data_to_save_property="avg_time",
            is_float_property=True,
            is_bold_min=True)

        with open(paths_config.latex_result_path, 'w') as outfile:
            outfile.write(latex_text)


def create_latex_table_string(result_directory, table_caption, data_to_save_property, is_float_property, is_bold_min):
    # results is a dictionary (key is graph name) of lists of DataToSave objects
    results = load_algorithm_run_data_from_directory(result_directory)
    sorted_graph_names = get_sorted_graph_names(results)
    sorted_algorithm_names = get_sorted_algorithm_names(results)

    latex_text = ""

    latex_text += create_latex_table_top(sorted_algorithm_names, table_caption)
    for graph_name in sorted_graph_names:
        latex_text += create_latex_table_row(
            graph_name, sorted_algorithm_names, results, data_to_save_property, is_float_property, is_bold_min)
    latex_text += create_latex_table_bottom()
    latex_text = latex_text.replace("_", "\\_")

    return latex_text


def create_legend(result_directory):
    results = load_algorithm_run_data_from_directory(result_directory)
    sorted_algorithm_names = get_sorted_algorithm_names(results)

    latex_text = "\\begin{itemize}\n"
    for i, algorithm_name in enumerate(sorted_algorithm_names):
        latex_text += "\\item A" + str(i) + " = " + algorithm_name + "\n"
    latex_text += "\\end{itemize}\n"

    return latex_text


def create_latex_table_row(graph_name, sorted_algorithm_names, results, data_to_save_property, is_float_property,
                           is_bold_min):
    latex_row = "{0} & {1} & {2:.2f}".format(
        results[graph_name][0]['graph_family'],
        results[graph_name][0]['graph_nr_of_vertices'],
        results[graph_name][0]['graph_density']
    )

    for algorithm_name in sorted_algorithm_names:
        algorithm_data_to_save = filter(lambda x: x['algorithm_name'] == algorithm_name, results[graph_name])[0]
        if is_float_property:
            latex_row += " & {0:.2f}".format(algorithm_data_to_save[data_to_save_property])
        else:
            latex_row += " & {0}".format(algorithm_data_to_save[data_to_save_property])

    latex_row = bold_extreme_value_latex_row(latex_row, 3, 3 + len(sorted_algorithm_names), is_bold_min)

    latex_row += " \\\\\n"
    return latex_row


def create_latex_table_top(sorted_algorithm_names, caption):
    longtable = "\\begin{longtable}[c]{|l|c|c|" + "c|" * len(sorted_algorithm_names) + "}\n"
    caption = "\\caption{" + caption + "}\\\\\n\\hline\n"
    header = "Graph & Vertices & Density"
    for i, algorithm in enumerate(sorted_algorithm_names):
        header += " & A" + str(i)
    header += "\\\\\n\\hline\n"

    return longtable + caption + header


def create_latex_table_bottom():
    return "\\end{longtable}\n"


def bold_extreme_value_latex_row(latex_row, from_column, to_column, is_bold_min):
    latex_row_values = latex_row.split(' & ')[from_column:to_column]
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
