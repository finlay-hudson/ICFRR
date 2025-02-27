import matplotlib.pyplot as plt
import numpy as np


def plot_results_over_iterations(n_times, maps_200s, maps_alls, prec_100s, prec_200s, save_plot="map_prec_ICFRR.png",
                                 every_x=5, every_x_ticks=5, title=None):
    xs = range(n_times + 1)
    print(f"max map@all was at iter {np.argmax(maps_alls)}")
    plt.plot(xs[::every_x], maps_alls[::every_x], label='map@all', color='blue', marker='o')
    plt.plot(xs[::every_x], maps_200s[::every_x], label='map@200', color='orange', marker='o')
    plt.plot(xs[::every_x], prec_100s[::every_x], label='prec@100', color='green', marker='s')
    plt.plot(xs[::every_x], prec_200s[::every_x], label='prec@200', color='red', marker='s')
    plt.xlabel('Num Iterations')
    plt.ylabel('Percentage')
    plt.xticks(xs[::every_x_ticks])
    if title is not None:
        plt.title(title)
    plt.legend()
    if save_plot:
        plt.savefig(save_plot)
    else:
        plt.show()
