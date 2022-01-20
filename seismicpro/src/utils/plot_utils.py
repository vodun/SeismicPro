"""Utilily functions for visualization"""

# pylint: disable=invalid-name
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def add_colorbar(ax, img, colorbar, divider=None):
    if not isinstance(colorbar, (bool, dict)):
        raise ValueError(f"colorbar must be bool or dict but {type(colorbar)} was passed")
    if colorbar is not False:
        colorbar = {} if colorbar is True else colorbar
        if divider is None:
            divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.figure.colorbar(img, cax=cax, **colorbar)


def set_ticks(ax, axis, axis_label, tick_labels, num=None, step_ticks=None, step_labels=None, round_to=0, **kwargs):
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
    locator, formatter = _process_ticks(labels=tick_labels, num=num, step_ticks=step_ticks, step_labels=step_labels,
                                        round_to=round_to)
    rotation_kwargs = _pop_rotation_kwargs(kwargs)
    ax_obj = getattr(ax, f"{axis}axis")
    ax_obj.set_label_text(axis_label, **kwargs)
    ax_obj.set_ticklabels([], **kwargs, **rotation_kwargs)
    ax_obj.set_major_locator(locator)
    ax_obj.set_major_formatter(formatter)


def _process_ticks(labels, num, step_ticks, step_labels, round_to):
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
        if (label_ix < 0) or (label_ix > len(labels) - 1):
            return None

        label_value = labels[np.round(label_ix).astype(np.int32)]
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
