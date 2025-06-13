import numpy as np
from matplotlib.lines import Line2D


def find_convergence_epoch(accuracies, window=10, threshold=1):
    for epoch in range(len(accuracies) - window + 1):
        window_data = accuracies[epoch:epoch + window]
        if np.ptp(window_data) <= threshold:
            return epoch + window
    return len(accuracies) - 1


def draw_convergence_marker(cur_ax, color, x, y, text, text_x, text_y):
    cur_ax.axvline(x, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    cur_ax.text(text_x, text_y, text, color=color, ha='center', va='top')
    cur_ax.scatter(x, y, marker='*', s=80, color=color, zorder=5)


def set_axes(cur_ax):
    custom_marker = [Line2D([], [], marker='*', linestyle='none', color='black', label='Convergence Point', markersize=9)]
    cur_ax.legend(handles=cur_ax.get_legend_handles_labels()[0] + custom_marker, loc='best')

    cur_ax.grid(linestyle=':')
    cur_ax.spines['top'].set_visible(False)
    cur_ax.spines['right'].set_visible(False)
