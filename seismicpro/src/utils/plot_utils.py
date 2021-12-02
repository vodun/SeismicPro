"""Utilily functions for visualization"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker


def save_figure(fig, path, dpi=100, bbox_inches="tight", pad_inches=0.1, **kwargs):
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)


def plot_arg_to_dict(arg):
    return arg.copy() if isinstance(arg, dict) else {"label": arg}


def set_text_formatting(kwargs):
    FORMAT_ARGS = {'fontsize', 'size', 'fontfamily', 'family', 'fontweight', 'weight'}
    TEXT_ARGS = {'title', 'x_ticker', 'y_ticker'}

    global_formatting = {arg: kwargs.pop(arg) for arg in FORMAT_ARGS if arg in kwargs}
    text_args = {arg: {**global_formatting, **plot_arg_to_dict(kwargs.pop(arg))} for arg in TEXT_ARGS if arg in kwargs}
    return {**kwargs, **text_args}


def set_ticks(ax, axis, axis_label, tick_labels, **kwargs):
    locator, formatter, kwargs = _process_ticks(labels=tick_labels, **kwargs)
    kwargs, rotation_kwargs = _process_kwargs(**kwargs)
    ax_obj = getattr(ax, f"{axis}axis")
    ax_obj.set_major_locator(locator)
    ax_obj.set_major_formatter(formatter)
    ax_obj.set_label_text(axis_label, **kwargs)
    getattr(plt, f"{axis}ticks")(**kwargs, **rotation_kwargs)


def _process_ticks(labels, num=None, step_ticks=None, step_labels=None, round_to=0, **kwargs):
    use_index = False
    n_labels = len(labels)
    locator = ticker.AutoLocator()

    if num is not None:
        locator = ticker.LinearLocator(num)
    elif step_ticks is not None:
        locator = ticker.IndexLocator(step_ticks, 0)
    elif step_labels is not None:
        if (np.diff(labels) < 0).any():
            raise ValueError("step_labels is valid only for monotonically increasing labels.")
        use_index = True
        candidates = np.arange((labels[0] // step_labels + 1) * step_labels, labels[-1], step_labels)
        ticks = np.concatenate([[0], np.searchsorted(labels, candidates), [n_labels - 1]])
        ticks, unique_indices = np.unique(ticks, return_index=True)
        locator = ticker.FixedLocator(ticks)
        labels = np.concatenate([[labels[0]], candidates, [labels[n_labels - 1]]])[unique_indices]

    def formatter(values, index):
        ix = index if use_index else values
        ix = int(np.clip(ix, 0, len(labels) - 1))
        sub_labels = labels[ix]

        if round_to is not None and sub_labels is not None:
                sub_labels = np.round(sub_labels, round_to)
                sub_labels = sub_labels.astype(int) if round_to == 0 else sub_labels
        return sub_labels

    return locator, formatter, kwargs


def _process_kwargs(**kwargs):
    ROTATION_ARGS = {"ha", "rotation_mode"}
    rotation = kwargs.pop("rotation", None)
    rotation_kwargs = {arg: kwargs.pop(arg) for arg in ROTATION_ARGS if arg in kwargs}
    if rotation is not None:
        rotation_kwargs = {"rotation": rotation, "ha": "right", "rotation_mode": "anchor", **rotation_kwargs}
    return kwargs, rotation_kwargs


def plot_metrics_map(metrics_map, cmap=None, title=None, figsize=(10, 7),  # pylint: disable=too-many-arguments
                     pad=False, fontsize=11, ticks_range_x=None, ticks_range_y=None,
                     x_ticker=None, y_ticker=None, save_to=None, dpi=300, **kwargs):
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

    x_ticker = {} if x_ticker is None else x_ticker
    y_ticker = {} if y_ticker is None else y_ticker
    set_ticks(ax, "x", None, np.linspace(*ticks_range_x, metrics_map.shape[1]), **x_ticker)
    set_ticks(ax, "y", None, np.linspace(*ticks_range_y, metrics_map.shape[0]), **y_ticker)

    if save_to:
        save_figure(fig, save_to, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.show()
