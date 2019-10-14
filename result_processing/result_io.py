import os
import re

from configuration import paths_config


def find_newest_result_seed():
    """Find the newest result seed from the names of subdirectories of base_directory assuming that the seeds have specific format

    :return: newest result seed from those found in base_directory with given format.
    """

    seed_pattern = re.compile("(.*)/\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2}")

    all_subdirs = filter(lambda d: seed_pattern.match(d), all_subdirs_of(paths_config.base_directory))
    newest_dir = max(all_subdirs, key=os.path.getmtime)
    split = newest_dir.split("/")
    return split[len(split) - 1]


def all_subdirs_of(b='.'):
    """Returns list of path to all subdirectories of b

    :param b: a directory
    :return: list of all subdirectories
    """

    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result
