from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("ggplot")

def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3

EPISODES = 70_000
SAVED_EVERY = 500
current = 0

fig = plt.figure(figsize=(12, 9))


for i in range (0, EPISODES+1, SAVED_EVERY):


    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)

    q_table = np.load(f"qtables/{70000}-qtable.py.npy")
    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            for k, k_vals in enumerate(y_vals):
                for j, j_vals in enumerate(k_vals):
                    ax1.scatter(k, j, c=get_q_color(j_vals[0], j_vals)[0], marker="o", alpha=get_q_color(j_vals[0], j_vals)[1])
                    ax2.scatter(k, j, c=get_q_color(j_vals[1], j_vals)[0], marker="o", alpha=get_q_color(j_vals[1], j_vals)[1])
                    ax1.set_ylabel("Action 0")
                    ax2.set_ylabel("Action 1")

    plt.savefig(f"qtable_charts/{i}.png")
    plt.clf()