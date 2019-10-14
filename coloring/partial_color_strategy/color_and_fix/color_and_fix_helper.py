from coloring.algorithm_helper import *


def is_better_partition(graph, partition1, partition2, independent_set_extraction_strategy):
    """ Checks whether partition1 is better than partition2 (partition is full, possibly illegal, coloring of graph)
        by comparing average number of nodes per color for legal partial coloring obtained from a partition. The result
        might depend on chosen strategy of obtaining legal partial coloring from partition.

    :param graph: (nx.Graph)
    :param partition1: (dict) Full, possibly illegal, coloring of graph
    :param partition2: (dict) Full, possibly illegal, coloring of graph
    :param independent_set_extraction_strategy: Determines strategy of obtaining legal partial coloring from a partition
    :return: True if partition1 is better than partition2
    """

    # TODO: When there are more hyperplanes it often chooses the resulting partition
    # TODO: as best even though it results in more colors (e.g. for DSJC 125.5)

    # if the partition2 is empty or None, partition1 is better
    if partition2 is None or len(partition2) == 0:
        return True

    # if partition2 is not empty but partition1 is, partition2 is better
    if partition1 is None or len(partition1) == 0:
        return False

    # for each partition, find nodes that need to be deleted from graph for the partition to be legal coloring
    nodes_to_delete1 = find_nodes_to_delete(graph, partition1,
                                            independent_set_extraction_strategy=independent_set_extraction_strategy)
    nodes_to_color1 = {n for n in graph.nodes() if n not in nodes_to_delete1}
    nr_of_colors1 = len(set(partition1.values()))

    nodes_to_delete2 = find_nodes_to_delete(graph, partition2,
                                            independent_set_extraction_strategy=independent_set_extraction_strategy)
    nodes_to_color2 = {n for n in graph.nodes() if n not in nodes_to_delete2}
    nr_of_colors2 = len(set(partition2.values()))

    # compute average number of nodes per color for each partition
    avg1 = float(len(nodes_to_color1)) / nr_of_colors1
    avg2 = float(len(nodes_to_color2)) / nr_of_colors2

    return avg1 > avg2
