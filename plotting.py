import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple

def subplots(rows = 1, cols = 1) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(rows, cols)
    return fig, ax

def time_label(ax: Axes, time: float):
    if ax.name == '3d':
        ax.text2D(1, 1, f"Time: {time:.2f}", transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right')
    else:
        ax.text(1, 1, f"Time: {time:.2f}", transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right')

def plot_func(z, Z, time, fig = None, ax = None, Z_min = 0, Z_max = 0) -> Figure:
    if fig is None and ax is None:
        fig, ax = subplots()
    ax.plot(z, Z)
    if Z_min != 0 and Z_max != 0:
        ax.set_ylim(min(Z_min, np.min(Z)), max(Z_max, np.max(Z)))
    time_label(ax, time)
    return fig

def plot_2D_colormap(X, Y, data, time, fig = None, ax = None):
    if fig is None and ax is None:
        fig, ax = subplots()
    contour_plot = ax.contourf(X, Y, data, 100)
    cbar = fig.colorbar(contour_plot, ax=ax)
    time_label(ax, time)
    return fig

def plot_2D_vectors(X, Y, U, V, time, fig = None, ax = None):
    if fig is None and ax is None:
        fig, ax = subplots()
    ax.quiver(X, Y, U, V)
    time_label(ax, time)
    return fig

def plot_3D_vectors(X, Y, Z, U, V, W, time, fig = None, ax = None):
    if fig is None and ax is None:
        fig, ax = subplots()
    ax.quiver(X, Y, Z, U, V, W)
    time_label(ax, time)
    return fig

