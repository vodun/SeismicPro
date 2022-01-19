"""Utilily functions for visualization"""

# pylint: disable=invalid-name
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors as mcolors
from scipy.interpolate import interp1d

def as_dict(val, key):
    """Construct a dict with a {`key`: `val`} structure if given `val` is not a `dict`, or copy `val` otherwise."""
    return val.copy() if isinstance(val, dict) else {key: val}


def save_figure(fig, fname, dpi=100, bbox_inches="tight", pad_inches=0.1, **kwargs):
    """Save the given figure. All `args` and `kwargs` are passed directly into `matplotlib.pyplot.savefig`."""
    fig.savefig(fname, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)


def set_text_formatting(kwargs):
    """Pop text formatting args from `kwargs` and set them as defaults for 'title', 'x_ticker' and 'y_ticker'."""
    FORMAT_ARGS = {'fontsize', 'size', 'fontfamily', 'family', 'fontweight', 'weight'}
    TEXT_ARGS = {'title', 'x_ticker', 'y_ticker'}

    global_formatting = {arg: kwargs.pop(arg) for arg in FORMAT_ARGS if arg in kwargs}
    text_args = {arg: {**global_formatting, **as_dict(kwargs.pop(arg), key="label")}
                 for arg in TEXT_ARGS if arg in kwargs}
    return {**kwargs, **text_args}


def set_ticks(ax, axis, axis_label, tick_labels, num=None, step_ticks=None,
              step_labels=None, round_to=0, tick_range=None, **kwargs):
    """Set ticks and labels for `x` or `y` axis depending on the `axis`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        An axis on which ticks are set.
    axis : "x" or "y"
        Whether to set ticks for "x" or "y" axis of `ax`.
    axis_label : str
        The label to set for `axis` axis.
    tick_labels : array-like
        An array of labels for axis ticks.
    num : int, optional, defaults to None
        The number of evenly spaced ticks on the axis.
    step_ticks : int, optional, defaults to None
        A step between two adjacent ticks in samples (e.g. place every hundredth tick).
    step_labels : int, optional, defaults to None
        A step between two adjacent tick in the units of the corresponding labels (e.g. place a tick every 200ms for an
        axis, whose labels are measured in milliseconds).
    round_to : int, optional, defaults to 0
        The number of decimal places to round tick labels to. If 0, tick labels will be cast to integers.
    kwargs : misc, optional
        Additional keyword arguments to control text formatting and rotation. Passed directly to
        `matplotlib.axis.Axis.set_label_text` and `matplotlib.axis.Axis.set_ticklabels`.
    """
    # Format axis label
    UNITS = {  # pylint: disable=invalid-name
        "time": " (ms)",
        "offset": " (m)",
    }
    axis_label += UNITS.get(axis_label, "")
    axis_label = str.capitalize(axis_label)

    ax_obj = getattr(ax, f"{axis}axis")
    # The object drawn can have single tick label (e.g., for single-trace `gather`) which leads to interp1d being
    # unable initiate, since both x and y should have at least 2 entries. Repeating this single label solves the issue.
    if len(tick_labels) == 1:
        tick_labels = np.repeat(tick_labels, 2)
    tick_range = ax_obj.get_data_interval() if tick_range is None else tick_range
    # matplotlib does not update data interval when new artist is redrawn on the existing axes in interactive mode,
    # which leads to incorrect tick position to label interpolation. To overcome this, call `ax.clear()` before drawing
    # a new artist.
    tick_interpolator = interp1d(np.linspace(*tick_range, len(tick_labels)), tick_labels,
                                 kind="nearest", bounds_error=False)
    locator, formatter = _process_ticks(labels=tick_labels, num=num, step_ticks=step_ticks, step_labels=step_labels,
                                        round_to=round_to, tick_interpolator=tick_interpolator)
    rotation_kwargs = _pop_rotation_kwargs(kwargs)
    ax_obj.set_label_text(axis_label, **kwargs)
    ax_obj.set_ticklabels([], **kwargs, **rotation_kwargs)
    ax_obj.set_major_locator(locator)
    ax_obj.set_major_formatter(formatter)


def _process_ticks(labels, num, step_ticks, step_labels, round_to, tick_interpolator):
    """Create an axis locator and formatter by given `labels` and tick layout parameters."""
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
        """Get tick label by its index in `labels` and format the resulting value."""
        _ = args

        label_value = tick_interpolator(label_ix)
        if np.isnan(label_value):
            return None
        if round_to is not None:
            label_value = np.round(label_value, round_to)
            label_value = label_value.astype(np.int32) if round_to == 0 else label_value
        return label_value

    return locator, formatter


def _pop_rotation_kwargs(kwargs):
    """Pop the keys responsible for text rotation from `kwargs`."""
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
        Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
    y_ticker : dict, optional, defaults to None
        Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
    save_to : str or dict, optional, defaults to None
        If `str`, a path to save the figure to.
        If `dict`, should contain keyword arguments to pass to `matplotlib.pyplot.savefig`. In this case, the path
        is stored under the `fname` key.
        Otherwise, the figure is not saved.
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
