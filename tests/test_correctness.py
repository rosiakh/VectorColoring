""" Module runs test on all graphs in test_instances folder using algorithms defined in test_algorithms. """

from datetime import datetime

import test_config
from graph.graph_io import *
from result_processing import result_io
from run.algorithm_runner import run_check_save_on_directory

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

    paths_config.run_seed = datetime.now().strftime("%m-%d_%H-%M-%S")
    for subdir in result_io.all_subdirs_of(test_config.test_graphs_directory):
        run_check_save_on_directory(test_config.test_algorithms, subdir)
