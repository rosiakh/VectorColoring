""" Module runs test on all graphs in test_instances folder using algorithms defined in test_algorithms. """

from datetime import datetime

import test_config
from graph.graph_io import *
from run.algorithm_runner import run_check_save_on_all_subdirectories

logging.basicConfig(format='%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

paths_config.run_seed = datetime.now().strftime("%m-%d_%H-%M-%S")
paths_config.run_seed = "08-17_10-28-34"
run_check_save_on_all_subdirectories(test_config.test_algorithms, test_config.test_graphs_directory)
