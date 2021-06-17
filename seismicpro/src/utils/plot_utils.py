"""Utilily functions for visualization"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def plot_metrics_map(metrics_map, cmap=None, title=None, figsize=(10, 7),  # pylint: disable=too-many-arguments
                     pad=False, fontsize=11, ticks_range_x=None, ticks_range_y=None,
                     x_ticks=15, y_ticks=15, save_to=None, dpi=300, **kwargs):
    """Plot a map with metric values.

    Parameters
    ----------
    metrics_map : array-like
        Array with aggregated metrics values.
    cmap : str or `~matplotlib.colors.Colormap`, optional
        `~matplotlib.imshow` colormap.
    title : str, optional
        The title of the plot.
    figsize : array-like with length 2, optional, defaults to (10, 7)
        Output figure size.
    pad : bool, optional, defaults to False
        If `True`, edges of the figure will be padded with a thin white line. Otherwise, the figure will remain
        unchanged.
    fontsize : int, optional, defaults to 11
        The size of the text on the plot.
    ticks_range_x : array-like with length 2, optional
        Min and max value of labels on the x-axis.
    ticks_range_y : array-like with length 2, optional
        Min and max value of labels on the y-axis.
    x_ticks : int, optional, defaults to 15
        The number of coordinates on the x-axis.
    y_ticks : int, optional, defaults to 15
        The number of coordinates on the y-axis.
    save_to : str, optional
        If given, save plot to the path specified.
    dpi : int, optional, defaults to 300
        Resolution for saved figure.
    kwargs : misc, optional
        Additional named arguments for :func:`matplotlib.pyplot.imshow`.

    Note
    ----
    1. The map is drawn with origin = 'lower' by default, keep it in mind when passing ticks_labels.
    """
    if cmap is None:
        colors = ((0.0, 0.6, 0.0), (.66, 1, 0), (0.9, 0.0, 0.0))
        cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors)
        cmap.set_under('black')
        cmap.set_over('red')

    origin = kwargs.pop('origin', 'lower')
    aspect = kwargs.pop('aspect', 'auto')
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(metrics_map, origin=origin, cmap=cmap, aspect=aspect, **kwargs)

    if pad:
        ax.use_sticky_edges = False
        ax.margins(x=0.01, y=0.01)

    ax.set_title(title, fontsize=fontsize)
    cbar = fig.colorbar(img, extend='both', ax=ax)
    cbar.ax.tick_params(labelsize=fontsize)

    set_ticks(ax=ax, img_shape=metrics_map.T.shape, ticks_range_x=ticks_range_x, ticks_range_y=ticks_range_y,
              x_ticks=x_ticks, y_ticks=y_ticks, fontsize=fontsize)

    if save_to:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def set_ticks(ax, img_shape, ticks_range_x=None, ticks_range_y=None, x_ticks=15, y_ticks=15, fontsize=None,
              rotation=45):
    """Set tick labels for x and y axes.

    Parameters
    ----------
    ax : matplotlib axes
        Axis to which ticks are set.
    img_shape : array with length 2
        Shape of the image to add ticks to.
    ticks_range_x : array-like with length 2, optional
        Min and max value of labels on the x-axis.
    ticks_range_y : array-like with length 2, optional
        Min and max value of labels on the y-axis.
    x_ticks : int, optional, defaults to 15
        The number of coordinates on the x-axis.
    y_ticks : int, optional, defaults to 15
        The number of coordinates on the y-axis.
    fontsize : int, optional
        The size of text.
    rotation : int, optional
        Degree of rotation of the labels on the x axis.
    """
    ax.set_xticks(np.linspace(0, img_shape[0]-1, x_ticks))
    ax.set_yticks(np.linspace(0, img_shape[1]-1, y_ticks))

    if ticks_range_x is not None:
        ticks_labels_x = np.linspace(*ticks_range_x, x_ticks).astype(np.int32)
        ax.set_xticklabels(ticks_labels_x, size=fontsize)
    if ticks_range_y is not None:
        ticks_labels_y = np.linspace(*ticks_range_y, y_ticks).astype(np.int32)
        ax.set_yticklabels(ticks_labels_y, size=fontsize)

    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right", rotation_mode="anchor")
