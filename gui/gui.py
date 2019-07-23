import Tkinter as tk
import logging
import random
from tkFileDialog import askopenfilename

from configuration.algorithm_instances_config import gui_algorithms_sorted
from graph import graph_io
from run import run_single

root = tk.Tk()
root.resizable(False, False)
root.wm_title("ColorApp")
root.iconbitmap("../resources/favicon.ico")

tk_path = tk.StringVar()
tk_algorithm_name = tk.StringVar()
tk_algorithm_name.set(random.choice(list([algorithm.get_algorithm_name() for algorithm in gui_algorithms_sorted])))

algorithm_results = None

logging.basicConfig(format='%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


def choose_graph_command():
    global color_graph_button
    global draw_graph_button
    tk_path.set(askopenfilename())
    color_graph_button.config(state=tk.NORMAL)
    # draw_graph_button.config(state=tk.NORMAL)
    logging.info("You've chosen a graph: {0}".format(tk_path.get()))


graph = None
coloring = None


def do_color_command():
    global algorithm_results
    algorithm_results = run_single.run_single(
        graph_path=tk_path.get(),
        algorithm=next(alg for alg in gui_algorithms_sorted if alg.get_algorithm_name() == tk_algorithm_name.get()))
    draw_graph_button.config(state=tk.NORMAL)


def draw_graph_command():
    global algorithm_results
    if algorithm_results is not None:
        graph = algorithm_results.keys()[0]
        coloring = algorithm_results[graph][0].best_coloring
        graph_io.draw_graph(graph=graph, colors=coloring, toConsole=False, toImage=True, filename=graph.name)


button_width = 20
button_height = 2
choose_graph_button = tk.Button(root, text='Choose graph', command=choose_graph_command,
                                width=button_width, height=button_height)
choose_graph_button.pack(pady=10)

color_graph_button = tk.Button(root, text='Color graph', state=tk.DISABLED, command=do_color_command,
                               width=button_width, height=button_height)
color_graph_button.pack(pady=10)

draw_graph_button = tk.Button(root, text='Draw graph', state=tk.DISABLED, command=draw_graph_command,
                              width=button_width, height=button_height)
draw_graph_button.pack(pady=10)

for algorithm in gui_algorithms_sorted:
    tk.Radiobutton(
        root,
        text=algorithm.get_algorithm_name(),
        pady=5, padx=20,
        variable=tk_algorithm_name,
        value=algorithm.get_algorithm_name()) \
        .pack(anchor=tk.W)

tk.mainloop()
