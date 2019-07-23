from coloring.algorithm_helper import *


def is_better_partition(graph, partition1, partition2, independent_set_extraction_strategy):
    """Checks whether the first partition is better than the second one."""

    # TODO: When there are more hyperplanes it often chooses the resulting partition
    # TODO: as best even though it results in more colors (e.g. for DSJC 125.5)

    if partition2 is None or len(partition2) == 0:
        return True

    if partition1 is None or len(partition1) == 0:
        return False

    # Remove colors from one endpoint of each illegal edge in each partition.
    nodes_to_delete1 = find_nodes_to_delete(graph, partition1, strategy=independent_set_extraction_strategy)
    nodes_to_color1 = {n for n in graph.nodes() if n not in nodes_to_delete1}
    nr_of_colors1 = len(set(partition1.values()))

    nodes_to_delete2 = find_nodes_to_delete(graph, partition2, strategy=independent_set_extraction_strategy)
    nodes_to_color2 = {n for n in graph.nodes() if n not in nodes_to_delete2}
    nr_of_colors2 = len(set(partition2.values()))

    avg1 = float(len(nodes_to_color1)) / nr_of_colors1
    avg2 = float(len(nodes_to_color2)) / nr_of_colors2

    return avg1 > avg2
