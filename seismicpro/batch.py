"""Implements SeismicBatch class for processing a small subset of seismic gathers"""

from string import Formatter
from functools import partial
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from batchflow import save_data_to, Batch, DatasetIndex, NamedExpression
from batchflow.decorators import action, inbatch_parallel

from .index import SeismicIndex
from .gather import Gather, CroppedGather
from .gather.utils.crop_utils import make_origins
from .semblance import Semblance, ResidualSemblance
from .field import Field
from .metrics import define_pipeline_metric, PartialMetric, MetricsAccumulator
from .decorators import create_batch_methods, apply_to_each_component
from .utils import to_list, as_dict, save_figure


@create_batch_methods(Gather, CroppedGather, Semblance, ResidualSemblance)
class SeismicBatch(Batch):
    """A batch class for seismic data that allows for joint and simultaneous processing of small subsets of seismic
    gathers in a parallel way.

    Initially, a batch contains unique identifiers of seismic gathers as its `index` and allows for their loading and
    processing. All the results are stored in batch attributes called `components` whose names are passed as `dst`
    argument of the called method.

    `SeismicBatch` implements almost no processing logic itself and usually just redirects method calls to objects in
    components specified in `src` argument. In order for a component method to be available in the batch, it should be
    decorated with :func:`~decorators.batch_method` in its class and the class itself should be listed in
    :func:`~decorators.create_batch_methods` decorator arguments of `SeismicBatch`.

    Examples
    --------
    Usually a batch is created from a `SeismicDataset` instance by calling :func:`~SeismicDataset.next_batch` method:
    >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
    >>> dataset = SeismicDataset(survey)
    >>> batch = dataset.next_batch(10)

    Here a batch of 10 gathers was created and can now be processed using the methods defined in
    :class:`~batch.SeismicBatch`. The batch does not contain any data yet and gather loading is usually the first
    method you want to call:
    >>> batch.load(src="survey")

    We've loaded gathers from a survey called `survey` in the component with the same name. Now the data can be
    accessed as a usual attribute:
    >>> batch.survey

    Almost all methods return a transformed batch allowing for method chaining:
    >>> batch.sort(src="survey", by="offset").plot(src="survey")

    Note that if `dst` attribute is omitted data processing is performed inplace.

    Parameters
    ----------
    index : DatasetIndex
        Unique identifiers of seismic gathers in the batch. Usually has :class:`~index.SeismicIndex` type.

    Attributes
    ----------
    index : DatasetIndex
        Unique identifiers of seismic gathers in the batch. Usually has :class:`~index.SeismicIndex` type.
    components : tuple of str or None
        Names of the created components. Each of them can be accessed as a usual attribute.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_calculated_metrics = 0

    def init_component(self, *args, dst=None, **kwargs):
        """Create and preallocate new attributes with names listed in `dst` if they don't exist and return ordinal
        numbers of batch items. This method is typically used as a default `init` function in `inbatch_parallel`
        decorator."""
        _ = args, kwargs
        dst = [] if dst is None else to_list(dst)
        for comp in dst:
            if self.components is None or comp not in self.components:
                self.add_components(comp, init=self.array_of_nones)
        return np.arange(len(self))

    @property
    def flat_indices(self):
        """np.ndarray: Unique identifiers of seismic gathers in the batch flattened into a 1d array."""
        if isinstance(self.index, SeismicIndex):
            return np.concatenate(self.indices)
        return self.indices

    @action
    def load(self, src=None, dst=None, fmt="sgy", combined=False, **kwargs):
        """Load seismic gathers into batch components.

        Parameters
        ----------
        src : str or list of str, optional
            Survey names to load gathers from.
        dst : str or list of str, optional
            Batch components to store the result in. Equals to `src` if not given.
        fmt : str, optional, defaults to "sgy"
            Data format to load gathers from.
        combined : bool, optional, defaults to False
            If `False`, load gathers by corresponding index value. If `True`, group all traces from a particular survey
            into a single gather. Increases loading speed by reducing the number of `DataFrame` indexations performed.
        kwargs : misc, optional
            Additional keyword arguments to :func:`~Survey.load_gather`.

        Returns
        -------
        batch : SeismicBatch
            A batch with loaded gathers. Creates or updates `dst` components inplace.

        Raises
        ------
        KeyError
            If unknown survey name was passed in `src`.
        """
        if isinstance(fmt, str) and fmt.lower() in {"sgy", "segy"}:
            if not combined:
                return self.load_gather(src=src, dst=dst, **kwargs)
            non_empty_parts = [i for i, n_gathers in enumerate(self.index.n_gathers_by_part) if n_gathers]
            combined_batch = type(self)(DatasetIndex(non_empty_parts), dataset=self.dataset, pipeline=self.pipeline)
            return combined_batch.load_combined_gather(src=src, dst=dst, parent_index=self.index, **kwargs)
        return super().load(src=src, fmt=fmt, dst=dst, **kwargs)

    @apply_to_each_component(target="threads", fetch_method_target=False)
    def load_gather(self, pos, src, dst, **kwargs):
        """Load a gather with ordinal number `pos` in the batch from a survey `src`."""
        index, part = self.index.index_by_pos(pos)
        getattr(self, dst)[pos] = self.index.get_gather(index, part=part, survey_name=src, **kwargs)

    @apply_to_each_component(target="for", fetch_method_target=False)
    def load_combined_gather(self, pos, src, dst, parent_index, **kwargs):
        """Load all batch traces from a given part and survey into a single gather."""
        part = parent_index.parts[self.indices[pos]]
        survey = part.surveys_dict[src]
        headers = part.headers.get(src, part.headers[[]])  # Handle the case when no headers were loaded for a survey
        getattr(self, dst)[pos] = survey.load_gather(headers, **kwargs)

    @action
    def update_field(self, field, src):
        """Update a field with objects from `src` component.

        Parameters
        ----------
        field : Field
            A field to update.
        src : str
            A component of instances to update the cube with. Each of them must have well-defined coordinates.

        Returns
        -------
        self : SeismicBatch
            The batch unchanged.
        """
        if not isinstance(field, Field):
            raise ValueError("Only a Field instance can be updated")
        field.update(getattr(self, src))
        return self

    @action
    def make_model_inputs(self, src, dst, mode='c', axis=0, expand_dims_axis=None):
        """Transform data to be used for model training.

        The method performs two-stage data processing:
        1. Stacks or concatenates input data depending on `mode` parameter along the specified `axis`,
        2. Inserts new axes to the resulting array at positions specified by `expand_dims_axis`.

        Source data to be transformed is passed to `src` argument either as an array-like of `np.ndarray`s or as a
        string, representing a name of batch component to get data from. Since this method is usually called in model
        training pipelines, `BA` named expression can be used to extract a certain attribute from each element of given
        component.

        Examples
        --------
        Given a dataset of individual traces, extract them from a batch of size 3 using `BA` named expression,
        concatenate into a single array, add a dummy axis and save the result into the `inputs` component:
        >>> pipeline = (Pipeline()
        ...     .load(src='survey')
        ...     .make_model_inputs(src=L('survey').data, dst='inputs', mode='c', axis=0, expand_dims_axis=1)
        ... )
        >>> batch = (dataset >> pipeline).next_batch(3)
        >>> batch.inputs.shape
        (3, 1, 1500)

        Parameters
        ----------
        src : src or array-like of np.ndarray
            Either a data to be processed itself or a component name to get it from.
        dst : src
            A component's name to store the combined result in.
        mode : {'c' or 's'}, optional, defaults to 'c'
            A mode that determines how to combine a sequence of arrays into a single one: 'c' stands for concatenating
            and 's' for stacking along the `axis`.
        axis : int or None, optional, defaults to 0
            An axis along which the arrays will be concatenated or stacked. If `mode` is `c`, `None` can be passed
            meaning that the arrays will be flattened before concatenation. Regardless of `mode`, `axis` must be no
            more than `data.ndim` - 1.
        expand_dims_axis : int or None, optional, defaults to None
            Insert new axes at the `expand_dims_axis` position in the expanded array. If `None`, the expansion does not
            occur.

        Returns
        -------
        self : SeismicBatch
            Batch with the resulting `np.ndarray` in the `dst` component.

        Raises
        ------
        ValueError
            If unknown `mode` was passed.
        """
        data = getattr(self, src) if isinstance(src, str) else src
        func = {'c': np.concatenate, 's': np.stack}.get(mode)
        if func is None:
            raise ValueError(f"Unknown mode '{mode}', must be either 'c' or 's'")
        data = func(data, axis=axis)

        if expand_dims_axis is not None:
            data = np.expand_dims(data, axis=expand_dims_axis)
        setattr(self, dst, data)
        return self

    @action(no_eval='dst')
    def split_model_outputs(self, src, dst, shapes):
        """Split data into multiple sub-arrays whose shapes along zero axis are defined by `shapes`.

        Usually gather data for each batch element is stacked or concatenated along zero axis using
        :func:`SeismicBatch.make_model_inputs` before being passed to a model. This method performs a reverse operation
        by splitting the received predictions allowing them to be matched with the corresponding batch elements for
        which they were obtained.

        Examples
        --------
        Given a dataset of individual traces, perform a segmentation model inference for a batch of size 3, split
        predictions and save them to the `outputs` batch component:
        >>> pipeline = (Pipeline()
        ...     .init_model(mode='dynamic', model_class=UNet, name='model', config=config)
        ...     .init_variable('predictions')
        ...     .load(src='survey')
        ...     .make_model_inputs(src=L('survey').data, dst='inputs', mode='c', axis=0, expand_dims_axis=1)
        ...     .predict_model('model', B('inputs'), fetches='predictions', save_to=B('predictions'))
        ...     .split_model_outputs(src='predictions', dst='outputs', shapes=L('survey').shape[0])
        ... )
        >>> batch = (dataset >> pipeline).next_batch(3)

        Each gather in the batch has shape (1, 1500), thus the created model inputs have shape (3, 1, 1500). Model
        predictions have the same shape as inputs:
        >>> batch.inputs.shape
        (3, 1, 1500)
        >>> batch.predictions.shape
        (3, 1, 1500)

        Predictions are split into 3 subarrays with a single trace in each of them to match the number of traces in the
        corresponding gathers:
        >>> len(batch.outputs)
        3
        >>> batch.outputs[0].shape
        (1, 1, 1500)

        Parameters
        ----------
        src : str or array-like of np.ndarray
            Either a data to be processed itself or a component name to get it from.
        dst : str or NamedExpression
            - If `str`, save the resulting sub-arrays into a batch component called `dst`,
            - If `NamedExpression`, save the resulting sub-arrays into the object described by named expression.
        shapes : 1d array-like
            An array with sizes of each sub-array along zero axis after the split. Its length should be generally equal
            to the current batch size and its sum must match the length of data defined by `src`.

        Returns
        -------
        self : SeismicBatch
            The batch with split data.

        Raises
        ------
        ValueError
            If data length does not match the sum of shapes passed.
            If `dst` is not of `str` or `NamedExpression` type.
        """
        data = getattr(self, src) if isinstance(src, str) else src
        shapes = np.cumsum(shapes)
        if shapes[-1] != len(data):
            raise ValueError("Data length must match the sum of shapes passed")
        split_data = np.split(data, shapes[:-1])

        if isinstance(dst, str):
            setattr(self, dst, split_data)
        elif isinstance(dst, NamedExpression):
            dst.set(value=split_data)
        else:
            raise ValueError(f"dst must be either `str` or `NamedExpression`, not {type(dst)}.")
        return self

    @action
    @inbatch_parallel(init='init_component', target='for')
    def crop(self, pos, src, origins, crop_shape, dst=None, joint=True, n_crops=1, stride=None, **kwargs):
        """Crop batch components.

        Parameters
        ----------
        src : str or list of str
            Components to be cropped. Objects in each of them must implement `crop` method which will be called from
            this method.
        origins : list, tuple, np.ndarray or str
            Origins define top-left corners for each crop or a rule used to calculate them. All array-like values are
            cast to an `np.ndarray` and treated as origins directly, except for a 2-element tuple of `int`, which will
            be treated as a single individual origin.
            If `str`, represents a mode to calculate origins. Two options are supported:
            - "random": calculate `n_crops` crops selected randomly using a uniform distribution over the source data,
              so that no crop crosses data boundaries,
            - "grid": calculate a deterministic uniform grid of origins, whose density is determined by `stride`.
        crop_shape : tuple with 2 elements
            Shape of the resulting crops.
        dst : str or list of str, optional, defaults to None
            Components to store cropped data. If `dst` is `None` cropping is performed inplace.
        joint : bool, optional, defaults to True
            Defines whether to create the same origins for all `src`s if passed `origins` is `str`. Generally used to
            perform joint random cropping of segmentation model input and output.
        n_crops : int, optional, defaults to 1
            The number of generated crops if `origins` is "random".
        stride : tuple with 2 elements, optional, defaults to crop_shape
            Steps between two adjacent crops along both axes if `origins` is "grid". The lower the value is, the more
            dense the grid of crops will be. An extra origin will always be placed so that the corresponding crop will
            fit in the very end of an axis to guarantee complete data coverage with crops regardless of passed
            `crop_shape` and `stride`.
        kwargs : misc, optional
            Additional keyword arguments to pass to `crop` method of the objects being cropped.

        Returns
        -------
        self : SeismicBatch
            The batch with cropped data.

        Raises
        ------
        TypeError
            If `joint` is `True` and `src` contains components of different types.
        ValueError
            If `src` and `dst` have different lengths.
            If `joint` is `True` and `src` contains components of different shapes.
        """
        dst = src if dst is None else dst
        src_list = to_list(src)
        dst_list = to_list(dst)

        if len(src_list) != len(dst_list):
            raise ValueError("src and dst should have the same length.")

        if joint:
            src_shapes = set()
            src_types = set()

            for src in src_list:  # pylint: disable=redefined-argument-from-local
                src_obj = getattr(self, src)[pos]
                src_types.add(type(src_obj))
                src_shapes.add(src_obj.shape)

            if len(src_types) > 1:
                raise TypeError("If joint is True, all src components must be of the same type.")
            if len(src_shapes) > 1:
                raise ValueError("If joint is True, all src components must have the same shape.")
            data_shape = src_shapes.pop()
            origins = make_origins(origins, data_shape, crop_shape, n_crops, stride)

        for src, dst in zip(src_list, dst_list):  # pylint: disable=redefined-argument-from-local
            src_obj = getattr(self, src)[pos]
            src_cropped = src_obj.crop(origins, crop_shape, n_crops, stride, **kwargs)
            setattr(self[pos], dst, src_cropped)

        return self

    @action(no_eval="save_to")
    def calculate_metric(self, metric, *args, metric_name=None, coords_component=None, save_to=None, **kwargs):
        """Calculate a metric for each batch element and store the results into an accumulator.

        The passed metric must be either a subclass of `PipelineMetric` or a `callable`. In the latter case, a new
        subclass of `PipelineMetric` is created with its `calc` method defined by the `callable`. The metric class is
        provided with information about the pipeline it was calculated in which allows restoring metric calculation
        context during interactive metric map plotting.

        Examples
        --------
        1. Calculate a metric, that estimates signal leakage after seismic processing by CDP gathers:

        Create a dataset with surveys before and after processing being merged:
        >>> header_index = ["INLINE_3D", "CROSSLINE_3D"]
        >>> header_cols = "offset"
        >>> survey_before = Survey(path_before, header_index=header_index, header_cols=header_cols, name="before")
        >>> survey_after = Survey(path_after, header_index=header_index, header_cols=header_cols, name="after")
        >>> dataset = SeismicDataset(survey_before, survey_after, mode="m")

        Iterate over the dataset and calculate the metric:
        >>> pipeline = (dataset
        ...     .pipeline()
        ...     .load(src=["before", "after"])
        ...     .calculate_metric(SignalLeakage, "before", "after", velocities=np.linspace(1500, 5500, 100),
        ...                       save_to=V("accumulator", mode="a"))
        ... )
        >>> pipeline.run(batch_size=16, n_epochs=1)

        Extract the created metric accumulator, construct the map and plot it:
        >>> leakage_map = pipeline.v("accumulator").construct_map()
        >>> leakage_map.plot(interactive=True)  # works only in JupyterLab with `%matplotlib widget` magic executed

        2. Calculate standard deviation of gather amplitudes using a lambda-function:
        >>> pipeline = (dataset
        ...     .pipeline()
        ...     .load(src="before")
        ...     .calculate_metric(lambda gather: gather.data.std(), "before", metric_name="std",
        ...                       save_to=V("accumulator", mode="a"))
        ... )
        >>> pipeline.run(batch_size=16, n_epochs=1)
        >>> std_map = pipeline.v("accumulator").construct_map()
        >>> std_map.plot(interactive=True, plot_component="before")

        Parameters
        ----------
        metric : subclass of PipelineMetric or callable
            The metric to calculate.
        metric_name : str or None, optional
            A name of the calculated metric. Obligatory if `metric` is `lambda` or `name` attribute is not overridden
            in the metric class.
        coords_component : str, optional
            A component name to extract coordinates from. If not given, the first argument passed to the metric
            calculation function is used.
        save_to : NamedExpression
            A named expression to save the constructed `MetricsAccumulator` instance to.
        args : misc, optional
            Additional positional arguments to the metric calculation function.
        kwargs : misc, optional
            Additional keyword arguments to the metric calculation function.

        Returns
        -------
        self : SeismicBatch
            The batch with increased `_num_calculated_metrics` counter.

        Raises
        ------
        ValueError
            If wrong type of `metric` is passed.
            If `metric` is `lambda` and `metric_name` is not given.
            If `metric` is a subclass of `PipelineMetric` and `metric.name` is `None`.
            If some batch item has `None` coordinates.
        """
        metric = define_pipeline_metric(metric, metric_name)
        unpacked_args, first_arg = metric.unpack_calc_args(self, *args, **kwargs)

        # Calculate metric values and their coordinates
        values = [metric.calc(*args, **kwargs) for args, kwargs in unpacked_args]
        coords_items = first_arg if coords_component is None else getattr(self, coords_component)
        coords = [item.coords for item in coords_items]
        if None in coords:
            raise ValueError("All batch items must have well-defined coordinates")

        # Construct a mapping from coordinates to ordinal numbers of gathers in the dataset index.
        # Later used by PipelineMetric to generate a batch by coordinates of a click on an interactive metric map.
        part_offsets = np.cumsum([0] + self.dataset.n_gathers_by_part[:-1])
        part_index_pos = [part.get_gathers_locs(indices) for part, indices in zip(self.dataset.parts, self.indices)]
        dataset_index_pos = np.concatenate([pos + offset for pos, offset in zip(part_index_pos, part_offsets)])
        coords_to_pos = defaultdict(list)
        for coord, pos in zip(coords, dataset_index_pos):
            coords_to_pos[tuple(coord)].append(pos)

        # Construct a metric and its accumulator
        metric = PartialMetric(metric, pipeline=self.pipeline, calculate_metric_index=self._num_calculated_metrics,
                               coords_to_pos=coords_to_pos)
        accumulator = MetricsAccumulator(coords, **{metric.name: {"values": values, "metric_type": metric}})

        if save_to is not None:
            save_data_to(data=accumulator, dst=save_to, batch=self)
        self._num_calculated_metrics += 1
        return self

    @staticmethod
    def _unpack_args(args, batch_item):
        """Replace all names of batch components in `args` with corresponding values from `batch_item`. """
        if not isinstance(args, (list, tuple, str)):
            return args

        unpacked_args = [getattr(batch_item, val) if isinstance(val, str) and val in batch_item.components else val
                         for val in to_list(args)]
        if isinstance(args, str):
            return unpacked_args[0]
        return unpacked_args

    @action
    def plot(self, src, src_kwargs=None, max_width=20, title="{src}: {index}", save_to=None, **common_kwargs):  # pylint: disable=too-many-statements
        """Plot batch components on a grid constructed as follows:
        1. If a single batch component is passed, its objects are plotted side by side on a single line.
        2. Otherwise, each batch element is drawn on a separate line, its components are plotted in the order they
           appear in `src`.

        If the total width of plots on a line exceeds `max_width`, the line is wrapped and the plots that did not fit
        are drawn below.

        This action calls `plot` methods of objects in components in `src`. There are two ways to pass arguments to
        these methods:
        1. `common_kwargs` set defaults for all of them,
        2. `src_kwargs` define specific `kwargs` for an individual component that override those in `common_kwargs`.

        Notes
        -----
        1. `kwargs` from `src_kwargs` take priority over the `common_kwargs` and `title` argument.
        2. `title` is processed differently than in the `plot` methods of objects in `src` components, see its
           description below for more details.

        Parameters
        ----------
        src : str or list of str
            Components to be plotted. Objects in each of them must implement `plot` method which will be called from
            this method.
        src_kwargs : dict or list of dicts, optional, defaults to None
            Additional arguments for plotters of components in `src`.
            If `dict`, defines a mapping from a component or a tuple of them to `plot` arguments, which are stored as
            `dict`s.
            If `list`, each element is a `dict` with arguments for the corresponding component in `src`.
        max_width : float, optional, defaults to 20
            Maximal figure width, measured in inches.
        title : str or dict, optional, defaults to "{src}: {index}"
            Title of subplots. If `dict`, should contain keyword arguments to pass to `matplotlib.axes.Axes.set_title`.
            In this case, the title string is stored under the `label` key.

            The title string may contain variables enclosed in curly braces that are formatted as python f-strings as
            follows:
            - "src" is substituted with the component name of the subplot,
            - "index" is substituted with the index of the current batch element,
            - All other variables are popped from the `title` `dict`.
        save_to : str or dict, optional, defaults to None
            If `str`, a path to save the figure to.
            If `dict`, should contain keyword arguments to pass to `matplotlib.pyplot.savefig`. In this case, the path
            is stored under the `fname` key.
            Otherwise, the figure is not saved.
        common_kwargs : misc, optional
            Additional common arguments to all plotters of components in `src`.

        Returns
        -------
        self : SeismicBatch
            The batch unchanged.

        Raises
        ------
        ValueError
            If the length of `src_kwargs` when passed as a list does not match the length of `src`.
            If any of the components' `plot` method is not decorated with `plotter` decorator.
        """
        # Construct a list of plot kwargs for each component in src
        src_list = to_list(src)
        if src_kwargs is None:
            src_kwargs = [{} for _ in range(len(src_list))]
        elif isinstance(src_kwargs, dict):
            src_kwargs = {src: src_kwargs[keys] for keys in src_kwargs for src in to_list(keys)}
            src_kwargs = [src_kwargs.get(src, {}) for src in src_list]
        else:
            src_kwargs = to_list(src_kwargs)
            if len(src_list) != len(src_kwargs):
                raise ValueError("The length of src_kwargs must match the length of src")

        # Construct a grid of plotters with shape (len(self), len(src_list)) for each of the subplots
        plotters = [[] for _ in range(len(self))]
        for src, kwargs in zip(src_list, src_kwargs):  # pylint: disable=redefined-argument-from-local
            # Merge src kwargs with common kwargs and defaults
            plotter_params = getattr(getattr(self, src)[0].plot, "method_params", {}).get("plotter")
            if plotter_params is None:
                raise ValueError("plot method of each component in src must be decorated with plotter")
            kwargs = {"figsize": plotter_params["figsize"], "title": title, **common_kwargs, **kwargs}

            # Scale subplot figsize if its width is greater than max_width
            width, height = kwargs.pop("figsize")
            if width > max_width:
                height = height * max_width / width
                width = max_width

            title_template = kwargs.pop("title")
            args_to_unpack = set(to_list(plotter_params["args_to_unpack"]))

            for i, index in enumerate(self.flat_indices):
                # Unpack required plotter arguments by getting the value of specified component with given index
                unpacked_args = {}
                for arg_name in args_to_unpack & kwargs.keys():
                    arg_val = kwargs[arg_name]
                    if isinstance(arg_val, dict) and arg_name in arg_val:
                        arg_val[arg_name] = self._unpack_args(arg_val[arg_name], self[i])
                    else:
                        arg_val = self._unpack_args(arg_val, self[i])
                    unpacked_args[arg_name] = arg_val

                # Format subplot title
                if title_template is not None:
                    src_title = as_dict(title_template, key='label')
                    label = src_title.pop("label")
                    format_names = {name for _, name, _, _ in Formatter().parse(label) if name is not None}
                    format_kwargs = {name: src_title.pop(name) for name in format_names if name in src_title}
                    src_title["label"] = label.format(src=src, index=index, **format_kwargs)
                    kwargs["title"] = src_title

                # Create subplotter config
                subplot_config = {
                    "plotter": partial(getattr(self, src)[i].plot, **{**kwargs, **unpacked_args}),
                    "height": height,
                    "width": width,
                }
                plotters[i].append(subplot_config)

        # Flatten all the subplots into a row if a single component was specified
        if len(src_list) == 1:
            plotters = [sum(plotters, [])]

        # Wrap lines of subplots wider than max_width
        split_pos = []
        curr_width = 0
        for i, plotter in enumerate(plotters[0]):
            curr_width += plotter["width"]
            if curr_width > max_width:
                split_pos.append(i)
                curr_width = plotter["width"]
        plotters = sum([np.split(plotters_row, split_pos) for plotters_row in plotters], [])

        # Define axes layout and perform plotting
        fig_width = max(sum(plotter["width"] for plotter in plotters_row) for plotters_row in plotters)
        row_heights = [max(plotter["height"] for plotter in plotters_row) for plotters_row in plotters]
        fig = plt.figure(figsize=(fig_width, sum(row_heights)), tight_layout=True)
        gridspecs = fig.add_gridspec(len(plotters), 1, height_ratios=row_heights)

        for gridspecs_row, plotters_row in zip(gridspecs, plotters):
            n_cols = len(plotters_row)
            col_widths = [plotter["width"] for plotter in plotters_row]

            # Create a dummy axis if row width is less than fig_width in order to avoid row stretching
            if fig_width > sum(col_widths):
                col_widths.append(fig_width - sum(col_widths))
                n_cols += 1

            # Create a gridspec for the current row
            gridspecs_col = gridspecs_row.subgridspec(1, n_cols, width_ratios=col_widths)
            for gridspec, plotter in zip(gridspecs_col, plotters_row):
                plotter["plotter"](ax=fig.add_subplot(gridspec))

        if save_to is not None:
            save_kwargs = as_dict(save_to, key="fname")
            save_figure(fig, **save_kwargs)
        return self
