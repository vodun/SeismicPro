"""Implements MetricMap class for metric visualization over a field map"""

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from .interactive_map import ScatterMapPlot, BinarizedMapPlot
from .utils import parse_coords, parse_index, parse_metric_data
from ..decorators import plotter
from ..utils import to_list, get_first_defined, add_colorbar, calculate_axis_limits, set_ticks, set_text_formatting


class BaseMetricMap:
    """Base metric map class. Implements general map data processing, visualization and reaggregation methods.

    Should not be instantiated directly, use `MetricMap` or its subclasses instead.
    """
    def __init__(self, coords, values, *, coords_cols=None, index=None, index_cols=None, metric=None, agg=None,
                 calculate_immediately=True, **context):
        coords, coords_cols = parse_coords(coords, coords_cols)
        values, metric = parse_metric_data(values, metric)
        metric_data = pd.DataFrame(coords, columns=coords_cols)
        metric_data[metric.name] = values

        if index is None:
            index_cols = coords_cols
        else:
            index, index_cols = parse_index(index, index_cols)
            metric_data[index_cols] = index

        if agg is None:
            default_agg = {True: "max", False: "min", None: "mean"}
            agg = default_agg[metric.is_lower_better]

        self._metric_data = None
        self._index_data = None
        self._map_data = None
        self._map_coords_to_indices = None
        self._index_to_coords = None

        self.metric_data_list = [metric_data]
        self.coords_cols = coords_cols
        self.index_cols = index_cols
        self.has_index = index is not None
        self.metric = metric
        self.bound_metric = None
        self.agg = agg
        self.context = context
        self.requires_recalculation = True
        if calculate_immediately:
            self._recalculate()

    @property
    def metric_name(self):
        return self.metric.name

    @property
    def metric_data(self):
        if self.requires_recalculation:
            self._recalculate()
        return self._metric_data

    @property
    def index_data(self):
        if self.requires_recalculation:
            self._recalculate()
        return self._index_data

    @property
    def map_data(self):
        if self.requires_recalculation:
            self._recalculate()
        return self._map_data

    @property
    def map_coords_to_indices(self):
        if self.requires_recalculation:
            self._recalculate()
        return self._map_coords_to_indices

    @property
    def index_to_coords(self):
        if self.requires_recalculation:
            self._recalculate()
        return self._index_to_coords

    @property
    def plot_title(self):
        """str: title of the map plot."""
        agg_name = self.agg.__name__ if callable(self.agg) else self.agg
        return f"{agg_name}({self.metric_name})"

    @property
    def x_tick_labels(self):
        """None or array-like: labels of x axis ticks."""
        return None

    @property
    def y_tick_labels(self):
        """None or array-like: labels of y axis ticks."""
        return None

    def append(self, other):
        if ((self.coords_cols != other.coords_cols) or
            (self.has_index is not other.has_index) or (self.index_cols != other.index_cols) or
            (type(self.metric) is not type(other.metric)) or (self.metric_name != other.metric_name)):
            raise ValueError("Only a map with the same types of coordinates, index and metric can be appended")
        self.metric_data_list += other.metric_data_list
        self.context.update(other.context)
        self.metric = other.metric
        self.bound_metric = None  # The context may have changed
        self.requires_recalculation = True

    def extend(self, other):
        self.append(other)

    def _recalculate(self):
        self._metric_data = pd.concat(self.metric_data_list, ignore_index=True, copy=False)
        requires_explode = pd.api.types.is_object_dtype(self._metric_data[self.metric_name].dtype)
        metric_agg = (lambda s: s.explode().agg(self.agg)) if requires_explode else self.agg
        agg_dict = {
            self.coords_cols[0]: (self.coords_cols[0], "first"),
            self.coords_cols[1]: (self.coords_cols[1], "first"),
            "_DELTA_X": (self.coords_cols[0], np.ptp),
            "_DELTA_Y": (self.coords_cols[1], np.ptp),
            self.metric_name: (self.metric_name, metric_agg),
        }
        index_data = self._metric_data.groupby(self.index_cols, as_index=False, sort=False).agg(**agg_dict)
        if index_data[["_DELTA_X", "_DELTA_Y"]].any(axis=None):
            raise ValueError
        index_data.drop(columns=["_DELTA_X", "_DELTA_Y"], inplace=True)
        self._index_data = index_data
        self._index_to_coords = index_data.groupby(self.index_cols)[self.coords_cols]
        self._calculate_map_data()
        self.requires_recalculation = False

    def _calculate_map_data(self):
        raise NotImplementedError

    def get_indices_by_map_coords(self, map_coords):
        return self.map_coords_to_indices.get_group(map_coords).set_index(self.index_cols)[self.metric_name]

    def get_coords_by_index(self, index):
        return tuple(self.index_to_coords.get_group(index).iloc[0].to_list())

    def evaluate(self, agg=None, use_global=False):
        """Aggregate metric values.

        Parameters
        ----------
        agg : str or callable, optional, defaults to None
            A function used for aggregating metric values. If not given, `agg` passed during map initialization is
            used. Passed directly to `pandas.core.groupby.DataFrameGroupBy.agg`.

        Returns
        -------
        metric_val : float
            Evaluated metric value.
        """
        if agg is None:
            agg = self.agg

        if use_global:
            metric_data = self.metric_data[self.metric_name]
            if pd.api.types.is_object_dtype(metric_data):
                metric_data = metric_data.explode()
            return metric_data.agg(agg)

        if agg == self.agg:
            return self.index_data[self.metric_name].agg(self.agg)
        return self.aggregate(agg=agg).index_data[self.metric_name].agg(agg)

    def aggregate(self, agg=None, bin_size=None):
        """Aggregate the map with new `agg` and `bin_size`.

        agg : str or callable, optional
            A function used for aggregating the map. If not given, will be determined by the value of `is_lower_better`
            attribute of the metric class in order to highlight outliers. Passed directly to
            `pandas.core.groupby.DataFrameGroupBy.agg`.
        bin_size : int, float or array-like with length 2, optional
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.

        Returns
        -------
        metrics_maps : BaseMetricMap
            Aggregated map.
        """
        return self.map_class(self.metric_data[self.coords_cols], self.metric_data[self.metric_name],
                              index=self.metric_data[self.index_cols], metric=self.metric, agg=agg, bin_size=bin_size,
                              **self.context)

    def get_worst_coords(self, is_lower_better=None):
        """Get coordinates with the worst metric value depending on `is_lower_better`. If not given, `is_lower_better`
        attribute of `self.metric` is used.

        Three options are possible:
        1. If `is_lower_better` is `True`, coordinates with maximum metric value are returned,
        2. If `is_lower_better` is `False`, coordinates with minimum metric value are returned,
        3. Otherwise, coordinates whose value has maximum absolute deviation from the mean metric value is returned.
        """
        is_lower_better = self.metric.is_lower_better if is_lower_better is None else is_lower_better
        if is_lower_better is None:
            return (self.map_data - self.map_data.mean()).abs().idxmax()
        if is_lower_better:
            return self.map_data.idxmax()
        return self.map_data.idxmin()

    def get_centered_norm(self, clip_threshold_quantile):
        """Return a matplotlib norm to center map data around its mean value."""
        global_mean = self.map_data.mean()
        clip_threshold = (self.map_data - global_mean).abs().quantile(clip_threshold_quantile)
        if np.isclose(clip_threshold, 0):
            clip_threshold = 0.1 if np.isclose(global_mean, 0) else 0.1 * abs(global_mean)
        return mcolors.CenteredNorm(global_mean, clip_threshold)

    @plotter(figsize=(10, 7))
    def _plot(self, *, title=None, x_ticker=None, y_ticker=None, is_lower_better=None, vmin=None, vmax=None, cmap=None,
              colorbar=True, center_colorbar=True, clip_threshold_quantile=0.95, keep_aspect=False, ax=None, **kwargs):
        """Plot the metric map."""
        is_lower_better = self.metric.is_lower_better if is_lower_better is None else is_lower_better
        vmin_vmax_passed = (vmin is not None) or (vmax is not None)
        vmin = get_first_defined(vmin, self.metric.vmin, self.metric.min_value)
        vmax = get_first_defined(vmax, self.metric.vmax, self.metric.max_value)

        if (not vmin_vmax_passed) and (is_lower_better is None) and center_colorbar:
            norm = self.get_centered_norm(clip_threshold_quantile)
        else:
            norm = mcolors.Normalize(vmin, vmax)

        if cmap is None:
            if is_lower_better is None:
                cmap = "coolwarm"
            else:
                colors = ((0.0, 0.6, 0.0), (.66, 1, 0), (0.9, 0.0, 0.0))
                if not is_lower_better:
                    colors = colors[::-1]
                cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)

        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)
        map_obj = self._plot_map(ax, is_lower_better=is_lower_better, cmap=cmap, norm=norm, **kwargs)
        ax.set_title(**{"label": self.plot_title, **title})
        ax.ticklabel_format(style="plain", useOffset=False)
        if keep_aspect:
            ax.set_aspect("equal", adjustable="box")
        add_colorbar(ax, map_obj, colorbar, y_ticker=y_ticker)
        set_ticks(ax, "x", self.coords_cols[0], self.x_tick_labels, **x_ticker)
        set_ticks(ax, "y", self.coords_cols[1], self.y_tick_labels, **y_ticker)

    def plot(self, *, interactive=False, plot_on_click=None, **kwargs):
        """Plot the metric map.

        Parameters
        ----------
        title : str, optional
            Map plot title. If not given, a default title with metric name, aggregation function and binarization info
            is shown.
        x_ticker : dict, optional
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticker : dict, optional
            Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
        is_lower_better : bool or None, optional
            Specifies if lower value of the metric is better. Affects the default colormap. Taken from `metric` if not
            given.
        vmin : float or None, optional
            Minimum colorbar value. Taken from `metric` if not given.
        vmax : float or None, optional
            Maximum colorbar value. Taken from `metric` if not given.
        cmap : str or matplotlib.colors.Colormap, optional
            Map colormap. If not given defined by `is_lower_better`: if it is `bool`, a green-red colormap is used,
            if `None` - "coolwarm".
        colorbar : bool or dict, optional, defaults to True
            Whether to add a colorbar to the right of the metric map plot. If `dict`, defines extra keyword arguments
            for `matplotlib.figure.Figure.colorbar`.
        center_colorbar : bool, optional, defaults to True
            Whether to center the colorbar around mean metric value if `is_lower_better` is `None`.
        clip_threshold_quantile : float, optional, defaults to 0.95
            Clip metric values on the colorbar whose deviation from metric mean is greater than that defined by the
            quantile. Has an effect only if `center_colorbar` is `True` and `is_lower_better` is `None`.
        keep_aspect : bool, optional, defaults to False
            Whether to keep aspect ratio of the map plot.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        kwargs : misc, optional
            Any additional arguments for `matplotlib.axes.Axes.scatter` or `matplotlib.axes.Axes.imshow` depending on
            the map type.
        interactive : bool, optional, defaults to `False`
            Whether to plot metric map in interactive mode. Clicking on the map will result in displaying the `views`
            defined by the map metric. If no views are implemented, `plot_on_click` argument must be specified. Note
            that if `Metric.get_views` expects any arguments, they must be passed to `kwargs`, see docs of the metric
            used for more details. Interactive plotting must be performed in a JupyterLab environment with the
            `%matplotlib widget` magic executed and `ipympl` and `ipywidgets` libraries installed.
        plot_on_click : callable or list of callable, optional, only for interactive mode
            Views called on each click to display some data representation at the click location. Each of them must
            accept click coordinates as a tuple as `coords` argument and axes to plot on as `ax` argument.
        plot_on_click_kwargs : dict or list of dict, optional, only for interactive mode
            Any additional arguments for each view.
        """
        if not interactive:
            return self._plot(**kwargs)

        if plot_on_click is None:
            if self.bound_metric is None:
                self.bound_metric = self.metric.bind_context(metric_map=self, **self.context)
            plot_on_click, kwargs = self.bound_metric.get_views(**kwargs)
        plot_on_click_list = to_list(plot_on_click)
        if len(plot_on_click_list) == 0:
            raise ValueError("At least one click view must be specified")
        return self.interactive_map_class(self, plot_on_click=plot_on_click_list, **kwargs).plot()


class ScatterMap(BaseMetricMap):
    """Construct a map by aggregating metric values defined for the same coordinates using `agg`. NaN values are
    ignored.

    Should not be instantiated directly, use `MetricMap` or its subclasses instead."""
    def __init__(self, *args, **kwargs):
        self._has_overlaying_indices = None
        super().__init__(*args, **kwargs)

    @property
    def has_overlaying_indices(self):
        if self.requires_recalculation:
            self._recalculate()
        return self._has_overlaying_indices

    def _calculate_map_data(self):
        self._map_coords_to_indices = self._index_data.groupby(self.coords_cols)
        self._has_overlaying_indices = (self._map_coords_to_indices.size() > 1).any()
        self._map_data = self._map_coords_to_indices[self.metric_name].agg(self.agg)

    def _plot_map(self, ax, is_lower_better, **kwargs):
        """Display map data as a scatter plot."""
        sort_key = None
        if is_lower_better is None:
            is_lower_better = True
            global_mean = self.map_data.mean()
            sort_key = lambda col: (col - global_mean).abs()  # pylint: disable=unnecessary-lambda-assignment
        # Guarantee that extreme values are always displayed on top of the others
        map_data = self.map_data.sort_values(ascending=is_lower_better, key=sort_key)
        coords_x, coords_y = map_data.index.to_frame().values.T
        ax.set_xlim(*calculate_axis_limits(coords_x))
        ax.set_ylim(*calculate_axis_limits(coords_y))
        return ax.scatter(coords_x, coords_y, c=map_data, **kwargs)


class BinarizedMap(BaseMetricMap):
    """Construct a binarized metric map.

    Binarization is performed in the following way:
    1. All stored coordinates are divided into bins of the given `bin_size`,
    2. All metric values are grouped by their bin,
    3. An aggregation is performed by calling `agg` for values in each bin. NaN values are ignored.

    Should not be instantiated directly, use `MetricMap` or its subclasses instead.
    """
    def __init__(self, *args, bin_size, **kwargs):
        if isinstance(bin_size, (int, float, np.number)):
            bin_size = (bin_size, bin_size)
        self.bin_size = np.array(bin_size)
        self.x_bin_coords = None
        self.y_bin_coords = None
        super().__init__(*args, **kwargs)

    def _calculate_map_data(self):
        # Perform a shallow copy of the grouped data since new columns are going to be appended
        map_data = self._index_data.copy(deep=False)

        # Binarize map coordinates
        bin_cols = ["BIN_X", "BIN_Y"]
        min_coords = map_data[self.coords_cols].min(axis=0).values
        map_data[bin_cols] = ((map_data[self.coords_cols] - min_coords) // self.bin_size).astype(int)
        x_bin_range = np.arange(map_data["BIN_X"].max() + 1)
        y_bin_range = np.arange(map_data["BIN_Y"].max() + 1)
        self.x_bin_coords = min_coords[0] + self.bin_size[0] * x_bin_range + self.bin_size[0] // 2
        self.y_bin_coords = min_coords[1] + self.bin_size[1] * y_bin_range + self.bin_size[1] // 2
        self._map_coords_to_indices = map_data.groupby(bin_cols)
        self._map_data = self._map_coords_to_indices[self.metric_name].agg(self.agg)

    @property
    def plot_title(self):
        """str: title of the map plot."""
        return super().plot_title + f" in {self.bin_size[0]}x{self.bin_size[1]} bins"

    @property
    def x_tick_labels(self):
        """array-like: labels of x axis ticks."""
        return self.x_bin_coords

    @property
    def y_tick_labels(self):
        """array-like: labels of y axis ticks."""
        return self.y_bin_coords

    def _plot_map(self, ax, is_lower_better, **kwargs):
        """Display map data as an image."""
        _ = is_lower_better

        # Construct an image of the map
        x = self.map_data.index.get_level_values(0)
        y = self.map_data.index.get_level_values(1)
        map_image = np.full((len(self.x_bin_coords), len(self.y_bin_coords)), fill_value=np.nan)
        map_image[x, y] = self.map_data

        kwargs = {"interpolation": "none", "origin": "lower", "aspect": "auto", **kwargs}
        return ax.imshow(map_image.T, **kwargs)


class MetricMapMeta(type):
    """A metric map metaclass that instantiates either a `scatter_map_class` or a `binarized_map_class` depending on
    whether `bin_size` was given."""
    def __call__(cls, *args, bin_size=None, **kwargs):
        if bin_size is None:
            map_class = cls.scatter_map_class
            interactive_map_class = cls.interactive_scatter_map_class
        else:
            map_class = cls.binarized_map_class
            interactive_map_class = cls.interactive_binarized_map_class
            kwargs["bin_size"] = bin_size

        instance = object.__new__(map_class)
        instance.__init__(*args, **kwargs)
        instance.map_class = cls
        instance.interactive_map_class = interactive_map_class
        return instance


class MetricMap(metaclass=MetricMapMeta):
    """Construct a map from metric values and their coordinates.

    Examples
    --------
    A map can be created directly from known values and coordinates:
    >>> metric_map = MetricMap(coords=[[0, 0], [0, 1], [1, 0], [1, 1]], metric_values=[1, 2, 3, 4])

    But usually maps are constructed via helper functions. One of the most common cases is to accumulate metric values
    in a pipeline and then convert them into a map:
    >>> survey = Survey(path, header_index="FieldRecord", header_cols=["SourceY", "SourceX", "offset"], name="raw")
    >>> dataset = SeismicDataset(survey)
    >>> pipeline = (dataset
    ...     .pipeline()
    ...     .load(src="raw")
    ...     .gather_metrics(MetricsAccumulator, coords=L("raw").coords, std=L("raw").data.std(),
    ...                     save_to=V("accumulator", mode="a"))
    ... )
    >>> pipeline.run(batch_size=16, n_epochs=1)
    >>> std_map = pipeline.v("accumulator").construct_map()

    The resulting map can be visualized by calling `plot` method:
    >>> std_map.plot()

    In case of a large number of points it makes sense to aggregate the map first to make the plot more clear:
    >>> std_map.aggregate(bin_size=100, agg="mean").plot()

    Parameters
    ----------
    coords : 2d array-like with 2 columns
        Metric coordinates for X and Y axes.
    metric_values : 1d array-like or array of 1d arrays
        One or more metric values for each pair of coordinates in `coords`. Must match `coords` in length.
    coords_cols : array-like with 2 elements, optional
        Names of X and Y coordinates. Usually names of survey headers used to extract coordinates from. Defaults to
        ("X", "Y") if not given and cannot be inferred from `coords`.
    metric : Metric or subclass of Metric, optional, defaults to Metric
        The metric whose values are used to construct the map.
    metric_name : str, optional
        Metric name. Defaults to "metric" if not given and cannot be inferred from `metric` and `metric_values`.
    agg : str or callable, optional
        A function used for aggregating the map. If not given, will be determined by the value of `is_lower_better`
        attribute of the metric class in order to highlight outliers. Passed directly to
        `pandas.core.groupby.DataFrameGroupBy.agg`.
    bin_size : int, float or array-like with length 2, optional
        Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.

    Attributes
    ----------
    metric_data : pandas.DataFrame
        A `DataFrame` with coordinates and metric values. NaN metric values are dropped.
    map_data : pandas.Series
        Aggregated map data. Series index stores either metric coordinates as is if `bin_size` was not given or indices
        of bins otherwise.
    coords_cols : array-like with 2 elements
        Names of X and Y coordinates.
    metric_name : str
        Name of the metric.
    metric : Metric or subclass of Metric
        A metric class or instance with `metric_name` and `coords_cols` attributes set.
    agg : str or callable
        A function used for aggregating the map.
    bin_size : 1d np.ndarray with 2 elements
        Bin size for X and Y axes. Available only if the map was binarized.
    """
    scatter_map_class = ScatterMap
    binarized_map_class = BinarizedMap
    interactive_scatter_map_class = ScatterMapPlot
    interactive_binarized_map_class = BinarizedMapPlot
