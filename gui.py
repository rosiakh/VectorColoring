import Tkinter as tk
import logging
import random
from tkFileDialog import askopenfilename

import algorithm_config
import run

root = tk.Tk()
root.resizable(False, False)
root.wm_title("ColorApp")
root.iconbitmap("favicon.ico")

tk_path = tk.StringVar()
tk_algorithm_name = tk.StringVar()
tk_algorithm_name.set(random.choice(list(algorithm_config.algorithms.keys())))

logging.basicConfig(format='%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


def choose_graph_command():
    global color_graph_button
    tk_path.set(askopenfilename())
    color_graph_button.config(state=tk.NORMAL)
    logging.info("You've chosen a graph: {0}".format(tk_path.get()))


def do_color_command():
    run.do_run(tk_path.get(), tk_algorithm_name.get())


button_width = 20
button_height = 2
choose_graph_button = tk.Button(root, text='Choose a graph', command=choose_graph_command,
                                width=button_width, height=button_height)
choose_graph_button.pack(pady=10)
color_graph_button = tk.Button(root, text='Color graph', state=tk.DISABLED, command=do_color_command,
                               width=button_width, height=button_height)
color_graph_button.pack(pady=10)

for it, (alg_name, alg_obj) in enumerate(algorithm_config.algorithms.iteritems()):
    tk.Radiobutton(root,
                   text=alg_name,
                   pady=5,
                   padx=20,
                   variable=tk_algorithm_name,
                   value=alg_name).pack(anchor=tk.W)

tk.mainloop()
