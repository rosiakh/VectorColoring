import os

base_directory = '/home/hubert/VectorColoring/'
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

run_seed = ''  # A String used to differentiate between runs of algorithm, set at run's start.


def vector_colorings_directory():
    directory = current_run_directory() + 'VectorColorings/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def logs_directory():
    directory = current_run_directory() + 'Logs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def current_run_directory():
    directory = base_directory + run_seed + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
