import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
import os
import numpy as np
import sys
import re
import pandas as pd

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def get_points(prefix, dist, n_points_list):
    no_partition_points_all = []
    accuracy_all = []

    for n_points in n_points_list:
        path = prefix + "/Accuracy_{dist}_{n_points}.log".format(dist=dist, n_points=n_points)
        no_partition_points = []
        with open(path, 'r') as fi:
            for line in fi:
                m = re.search(r"No partition, Level: \d+, Index: \d+, Points: (\d+),", line)
                if m is not None:
                    no_partition_points.append(int(m.groups()[0]))
                m = re.search(r"window_query accuracy: (.*?)$", line)
                if m is not None:
                    accuracy_all.append(float(m.groups()[0]))
        no_partition_points_all.append(gini(np.asarray(no_partition_points)))
    return np.asarray(no_partition_points_all), np.asarray(accuracy_all)


def scale_size(size_list, k_scale=1024):
    return tuple(str(int(kb)) + "K" if kb < k_scale else str(int(kb / k_scale)) + "M" for kb in
                 np.asarray(size_list) / k_scale)


def draw_query_size(prefix, ):
    titles = ("Gini", "Accuracy")
    query_size_list = (1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000)
    loc = [x for x in range(len(query_size_list))]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 3.5))

    gini_all, accuracy_all = get_points(prefix, "uniform", query_size_list)
    ax_gini = axes[0]
    ax_accuracy = axes[1]

    ax_gini.plot(loc, gini_all, marker='x', label='Gini')
    ax_accuracy.plot(loc, accuracy_all, marker='x', label='Accuracy')
    # ax.set_yscale('log')

    for i, ax in zip(range(len(axes)), axes):
        ax.set_xticks(loc, scale_size(query_size_list, 1000), rotation=0)
        ax.legend(loc='upper left', ncol=2, handletextpad=0.2, columnspacing=0.8,
                  fontsize='medium', borderaxespad=1, borderpad=0, frameon=False)
        ax.set_title(titles[i], verticalalignment="top")
        ax.autoscale(tight=True)
        ax.margins(x=0.1, y=0.5)
        ax.set_xlabel(xlabel='# of Points')
        if i == 0:
            ax.set_ylabel(ylabel='Gini coefficient', labelpad=1)
        else:
            ax.set_ylabel(ylabel='Accuracy', labelpad=1)
    # # ylim = list(ax.get_ylim())
    # # ylim[0] = 0
    # # ylim[1] *= 1.5
    # # if i !=
    # # ax.set_ylim(ylim)


    fig.tight_layout()
    fig.savefig(os.path.join(prefix, '../', 'distribution.pdf'), format='pdf',
                bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dir = os.path.dirname(sys.argv[0])
    prefix = os.path.join(dir, "accuracy/")
    draw_query_size(prefix, )
