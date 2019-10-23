def is_better_ind_sets(graph, ind_sets1, ind_sets2):
    """Returns true if the first list of independent sets is better than the second.

    :param graph: (nx.Graph) unused
    :param ind_sets1: list of independent sets of vertices
    :param ind_sets2: list of independent sets of vertices
    :return true iff ind_sets1 is better than ind_sets2
    """

    if ind_sets2 is None or len(ind_sets2) == 0:
        return True

    if ind_sets1 is None or len(ind_sets1) == 0:
        return False

    # take set-theoretical sum of vertices of a list of independent sets
    temp_colors1 = {v: -1 for s in ind_sets1 for v in s}
    temp_colors2 = {v: -1 for s in ind_sets2 for v in s}

    if len(set(temp_colors2.values())) == 0:
        return False

    if len(set(temp_colors1.values())) == 0:
        return False

    # color each independent set with different color
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

    # calculate the score for list of independet sets by computing average number vertices colored with one color
    # this way of computing the score is not optimal as there may be some set that is completely contained in sum of
    # other sets and only by coloring it first, will we spare its color
    avg1 = len(temp_colors1.keys()) / float(len(set(temp_colors1.values())))
    avg2 = len(temp_colors2.keys()) / float(len(set(temp_colors2.values())))

    return avg1 > avg2
