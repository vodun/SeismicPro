"""Utilily functions for visualization"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors as mcolors


def plot_metrics_map(metrics_map, cmap=None, title=None, figsize=(10, 7),  # pylint: disable=too-many-arguments
                     pad=False, fontsize=11, ticks_range_x=None, ticks_range_y=None,
                     x_ticks=15, y_ticks=15, save_to=None, dpi=300, **kwargs):
    """Plot a map with metric values.

    Notes
    -----
    The map is drawn with `origin='lower'` by default, keep it in mind when passing arguments, related to axes ticks.

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
        The resolution of saved figure in dots per inch.
    kwargs : misc, optional
        Additional named arguments for :func:`matplotlib.pyplot.imshow`.
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
    ax : matplotlib.Axes
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
    rotation : int, optional, defaults to 45
        Rotation angle of the labels on the x axis. Measured in degrees.
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


def get_ticklabels(*tickers):
    tickers_list = []
    for ticker in tickers:
        tickers_list.append(ticker.pop('label', None) if isinstance(ticker, dict) else ticker)
    if len(tickers_list) < 2:
        return tickers_list[0]
    return tuple(tickers_list)


def set_ticks_and_labels(ax, shape, x_label=None, x_ticklabels=None, y_label=None, y_ticklabels=None, x_kwargs=None,
                         y_kwargs=None):
    x_ticklabels, x_ticks, x_rotation, x_kwargs = _process_ticks(x_ticklabels, shape[0], x_kwargs)
    x_ticklabels = x_ticks if x_ticklabels is None else x_ticklabels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, **x_kwargs)
    if x_rotation:
        plt.setp(ax.get_xticklabels(), **x_rotation)

    y_ticklabels, y_ticks, y_rotation, y_kwargs = _process_ticks(y_ticklabels, shape[1], y_kwargs)
    y_ticklabels = y_ticks if y_ticklabels is None else y_ticklabels
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, **y_kwargs)
    if y_rotation:
        plt.setp(ax.get_yticklabels(), **y_rotation)

    ax.set_xlabel(x_label, **x_kwargs)
    ax.set_ylabel(y_label, **y_kwargs)


def _process_ticks(labels, length, kwargs):
    rotation = False
    ticklabels = None
    kwargs = kwargs if isinstance(kwargs, dict) else {}
    num = kwargs.pop('num', None)
    step = kwargs.pop('step', None)
    round_to = kwargs.pop('round', int)
    rotation = kwargs.pop('rotation', None)
    num = 10 if num is None and step is None else num
    if num is not None:
        ticks = np.linspace(0, length-1, num=num)
    else:
        ticks = np.arange(0, length)[::step]
    ticks = ticks.astype(int)

    if labels is not None:
        if len(labels) != length:
            raise ValueError(f'Given {labels} `labels` while exepected {length}')
        ticklabels = labels[ticks]

        if round_to is not None:
            if isinstance(round_to, int):
                ticklabels = np.round(ticklabels, round_to)
            elif isinstance(round_to, type):
                ticklabels = ticklabels.astype(round_to)

    if rotation is not None:
        if not isinstance(rotation, dict):
            rotation = {
                "rotation": rotation,
                "ha": "right",
                "rotation_mode": "anchor"
            }
    return ticklabels, ticks, rotation, kwargs


def fill_text_kwargs(kwargs):
    TEXT_KEYS = ['fontsize', 'size', 'fontfamily', 'family', 'fontweight', 'weight']
    TEXT_ARGS = ['title', 'x_ticker', 'y_ticker']
    for key in TEXT_KEYS:
        item = kwargs.pop(key, None)
        if item is not None:
            for args in TEXT_ARGS:
                params = kwargs.get(args, None)
                if not isinstance(params, dict):
                    kwargs[args] = {'label': params, key: item}
    return kwargs


def save_figure(path, dpi, **kwargs):
    """!!!"""
    save_kwargs = {
        'bbox_inches': 'tight',
        'pad_inches': 0.1
    }
    save_kwargs.update(kwargs)
    plt.savefig(path, dpi=dpi, **save_kwargs)
