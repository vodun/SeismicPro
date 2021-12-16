"""Utilily functions for visualization"""

# pylint: disable=invalid-name
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors as mcolors


def as_dict(val, key):
    """Construct a dict with a structure {`key`: `val`} if given `val` is not dict, or copy `val` otherwise"""
    return val.copy() if isinstance(val, dict) else {key: val}


def save_figure(fig, fname, dpi=100, bbox_inches="tight", pad_inches=0.1, **kwargs):
    """Save the given figure `fig`. All arguemnts and `kwargs` are directly passed into `matplotlib.pyplot.savefig`."""
    fig.savefig(fname, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)


def set_text_formatting(kwargs):
    """Pop text related arguments from `kwargs` and add them to the following keys: 'title', 'x_ticker', 'y_ticker'"""
    FORMAT_ARGS = {'fontsize', 'size', 'fontfamily', 'family', 'fontweight', 'weight'}
    TEXT_ARGS = {'title', 'x_ticker', 'y_ticker'}

    global_formatting = {arg: kwargs.pop(arg) for arg in FORMAT_ARGS if arg in kwargs}
    text_args = {arg: {**global_formatting, **as_dict(kwargs.pop(arg), key="label")}
                       for arg in TEXT_ARGS if arg in kwargs}
    return {**kwargs, **text_args}


def set_ticks(ax, axis, axis_label, tick_labels, num=None, step_ticks=None, step_labels=None, round_to=0, **kwargs):
    """Set ticks and ticklabels for x or y axis depending on the "axis".

    Parameters
    ----------
    ax : matplotlib.Axes
        Axis to which ticks are set.
    axis : "x" or "y"
        Whether to set ticks for "x" or "y" axis.
    axis_label : str
        The label of the current axis.
    tick_labels : array-like
        Array of ticklabels.
    num : int, optional, defaults to None
        Number of ticks on the axis that are evenly spaced.
    step_ticks : int, optional, defaults to None
        Step between ticks. Ticks are placed evenly with a step equal to `step_ticks`.
    step_labels : int, optional, defaults to None
        Step between ticks. Ticks are placed at an exact distance by ticklabels.
    round_to : int, optional, defaults to 0
        Number of decimal places to round to. If `round_to` is 0, labels cast to an integer.
    kwargs : misc, optional
        Additional keyword arguments to control text formatting and rotation. Passes directly to
        `matplotlib.axis.Axis.set_label_text` and `matplotlib.axis.Axis.set_ticklabels`.
    """
    locator, formatter = _process_ticks(labels=tick_labels, num=num, step_ticks=step_ticks, step_labels=step_labels,
                                        round_to=round_to)
    rotation_kwargs = _pop_rotation_kwargs(kwargs)
    ax_obj = getattr(ax, f"{axis}axis")
    ax_obj.set_label_text(axis_label, **kwargs)
    ax_obj.set_ticklabels([], **kwargs, **rotation_kwargs)
    ax_obj.set_major_locator(locator)
    ax_obj.set_major_formatter(formatter)


def _process_ticks(labels, num, step_ticks, step_labels, round_to):
    """Create locator and formatter based on `labels` and desired tick steps"""
    if num is not None:
        locator = ticker.LinearLocator(num)
    elif step_ticks is not None:
        locator = ticker.IndexLocator(step_ticks, 0)
    elif step_labels is not None:
        if (np.diff(labels) < 0).any():
            raise ValueError("step_labels is valid only for monotonically increasing labels.")
        candidates = np.arange(labels[0], labels[-1], step_labels)
        ticks = np.searchsorted(labels, candidates)
        # Always include last label along the axis and remove duplicates
        ticks = np.unique(np.append(ticks, len(labels) - 1))
        locator = ticker.FixedLocator(ticks)
    else:
        locator = ticker.AutoLocator()

    def formatter(label_ix, *args):
        """Get label value for given label index in `label_ix`"""
        _ = args
        if (label_ix < 0) or (label_ix > len(labels) - 1):
            return None

        label_value = labels[np.round(label_ix).astype(np.int32)]
        if round_to is not None:
            label_value = np.round(label_value, round_to)
            label_value = label_value.astype(np.int32) if round_to == 0 else label_value
        return label_value

    return locator, formatter


def _pop_rotation_kwargs(kwargs):
    """Pop keys obliged for text rotation"""
    ROTATION_ARGS = {"ha", "rotation_mode"}
    rotation = kwargs.pop("rotation", None)
    rotation_kwargs = {arg: kwargs.pop(arg) for arg in ROTATION_ARGS if arg in kwargs}
    if rotation is not None:
        rotation_kwargs = {"rotation": rotation, "ha": "right", "rotation_mode": "anchor", **rotation_kwargs}
    return rotation_kwargs


def plot_metrics_map(metrics_map, cmap=None, title=None, figsize=(10, 7),  # pylint: disable=too-many-arguments
                     pad=False, fontsize=11, ticks_range_x=None, ticks_range_y=None,
                     x_ticker=None, y_ticker=None, save_to=None, **kwargs):
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
    ticks_range_x : array-like with length 2, optional, defaults to None
        Min and max value of labels on the x-axis.
    ticks_range_y : array-like with length 2, optional, defaults to None
        Min and max value of labels on the y-axis.
    x_ticker : dict, optional, defaults to None
        Paramters for ticks and ticklabels formatting for the x-aixs; see `.utils.set_ticks` for more details.
    y_ticker : dict, optional, defaults to None
        Paramters for ticks and ticklabels formatting for the y-aixs; see `.utils.set_ticks` for more details.
    save_to : str or dict, optional, defaults to None
        If `str`, a path to the resulting figure. Otherwise, all the `kwargs` to `matplotlib.pyplot.savefig`.
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

    if save_to is not None:
        save_kwargs = as_dict(save_to, key="fname")
        save_figure(fig, **save_kwargs)
    plt.show()
