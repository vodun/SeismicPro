"""Implements SeismicBatch class for processing a small subset of seismic gathers"""

from string import Formatter
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from .gather import Gather
from .cropped_gather import CroppedGather
from .semblance import Semblance, ResidualSemblance
from .metrics import define_metric, PipelineMetric, MetricAccumulator
from .decorators import create_batch_methods, apply_to_each_component
from .utils import to_list, as_dict, save_figure, make_origins
from ..batchflow import action, inbatch_parallel, save_data_to, Batch, DatasetIndex, NamedExpression


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
    >>> dataset = SeismicDataset(surveys=survey)
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
    components : tuple
        Names of the created components. Each of them can be accessed as a usual attribute.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calculated_metrics = 0

    @property
    def nested_indices(self):
        """list: indices of the batch each additionally wrapped into a list. If used as an `init` function in
        `inbatch_parallel` decorator, each index will be passed to its parallelly executed callable as a tuple, not
        individual level values."""
        if isinstance(self.indices, np.ndarray):
            return self.indices.tolist()
        return [[index] for index in self.indices]

    def _init_component(self, *args, dst=None, **kwargs):
        """Create and preallocate new attributes with names listed in `dst` if they don't exist and return
        `self.nested_indices`. This method is typically used as a default `init` function in `inbatch_parallel`
        decorator."""
        _ = args, kwargs
        dst = [] if dst is None else to_list(dst)
        for comp in dst:
            if self.components is None or comp not in self.components:
                self.add_components(comp, init=self.array_of_nones)
        return self.nested_indices

    @action
    def load(self, src=None, fmt="sgy", components=None, combined=False, **kwargs):
        """Load seismic gathers into batch components.

        Parameters
        ----------
        src : str or list of str, optional
            Survey names to load gathers from.
        fmt : str, optional, defaults to "sgy"
            Data format to load gathers from.
        components : str or list of str, optional
            Batch components to store the result in. Equals to `src` if not given.
        combined : bool, optional, defaults to False
            If `False`, load gathers by corresponding index value. If `True`, group all batch traces from a particular
            survey into a single gather increasing loading speed by reducing the number of `.loc`s performed.
        kwargs : misc, optional
            Additional keyword arguments to :func:`~Survey.load_gather`.

        Returns
        -------
        batch : SeismicBatch
            A batch with loaded gathers. Creates or updates `src` components inplace.

        Raises
        ------
        KeyError
            If unknown survey name was passed in `src`.
        """
        if isinstance(fmt, str) and fmt.lower() in {"sgy", "segy"}:
            if not combined:
                return self._load_gather(src=src, dst=components, **kwargs)
            unique_files = self.indices.unique(level=0)
            combined_batch = type(self)(DatasetIndex(unique_files), dataset=self.dataset, pipeline=self.pipeline)
            # pylint: disable=protected-access
            return combined_batch._load_combined_gather(src=src, dst=components, parent_index=self.index, **kwargs)
            # pylint: enable=protected-access
        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    @apply_to_each_component(target="for", fetch_method_target=False)
    def _load_gather(self, index, src, dst, **kwargs):
        """Load a gather with given `index` from a survey called `src`."""
        pos = self.index.get_pos(index)
        concat_id, gather_index = index[0], index[1:]
        # Unpack tuple in case of non-multiindex survey
        if len(gather_index) == 1:
            gather_index = gather_index[0]
        # Guarantee, that a DataFrame is always returned after .loc, regardless of pandas behaviour
        gather_index = slice(gather_index, gather_index)
        getattr(self, dst)[pos] = self.index.get_gather(survey_name=src, concat_id=concat_id,
                                                        gather_index=gather_index, **kwargs)

    @apply_to_each_component(target="for", fetch_method_target=False)
    def _load_combined_gather(self, index, src, dst, parent_index, **kwargs):
        """Load all batch traces from a survey called `src` with `CONCAT_ID` equal to `index` into a single gather."""
        pos = self.index.get_pos(index)
        gather_index = parent_index.indices.to_frame().loc[index].index
        getattr(self, dst)[pos] = parent_index.get_gather(survey_name=src, concat_id=index,
                                                          gather_index=gather_index, **kwargs)

    @action
    def update_velocity_cube(self, velocity_cube, src):
        """Update a velocity cube with stacking velocities from `src` component.

        Notes
        -----
        All passed `StackingVelocity` instances must have not-None coordinates.

        Parameters
        ----------
        velocity_cube : VelocityCube
            A cube to update.
        src : str
            A component with stacking velocities to update the cube with.

        Returns
        -------
        self : SeismicBatch
            The batch unchanged.

        Raises
        ------
        TypeError
            If wrong type of stacking velocities was passed.
        ValueError
            If any of the passed stacking velocities has `None` coordinates.
        """
        velocity_cube.update(getattr(self, src))
        return self

    @action
    def make_model_inputs(self, src, dst, mode='c', axis=0, expand_dims_axis=None):
        """Transform data to be used for model training.

        The method performes two-stage data processing:
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
        """Split data into multiple sub-arrays whose shapes along zero axis if defined by `shapes`.

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

        Predictions are split into 3 subarrays with a signle trace in each of them to match the number of traces in the
        correponding gathers:
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
    @inbatch_parallel(init='_init_component', target='for')
    def crop(self, idx, src, origins, crop_shape, dst=None, joint=True, n_crops=1, stride=None, **kwargs):
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

        pos = self.index.get_pos(idx)

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

    def _define_metric(self, metric, metric_name):
        is_metric_type = isinstance(metric, type) and issubclass(metric, PipelineMetric)
        is_callable = not isinstance(metric, type) and callable(metric)
        if not (is_metric_type or is_callable):
            raise ValueError("metric must be either a subclass of PipelineMetric or a callable "
                             f"but {type(metric)} given")
        if is_callable:
            metric_name = metric_name or metric.__name__
            if metric_name == "<lambda>":
                raise ValueError("metric_name must be passed for lambda metrics")
            metric = define_metric(base_cls=PipelineMetric, name=metric_name, calc=staticmethod(metric))
        else:
            metric_name = metric_name or metric.name
        return metric, metric_name

    @action(no_eval="save_to")
    def calculate_metric(self, metric, *args, metric_name=None, coords_component=None, coords_cols="auto",
                         save_to=None, **kwargs):
        metric, metric_name = self._define_metric(metric, metric_name)
        unpacked_args, first_arg = metric.unpack_calc_args(self, *args, **kwargs)

        coords_items = first_arg if coords_component is None else getattr(self, coords_component)
        coords = [item.get_coords(coords_cols) for item in coords_items]
        metric_params = {
            "values": [metric.calc(*args, **kwargs) for args, kwargs in unpacked_args],
            "metric_type": metric,
            "pipeline": self.pipeline,
            "calculate_metric_index": self._calculated_metrics,
        }
        accumulator = MetricAccumulator(coords, indices=self.indices, **{metric_name: metric_params})

        if save_to is not None:
            save_data_to(data=accumulator, dst=save_to, batch=self)
        self._calculated_metrics += 1
        return self

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
        # Consturct a list of plot kwargs for each component in src
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

            for i, index in enumerate(self.indices):
                # Unpack required plotter arguments by getting the value of specified component with given index
                unpacked_args = {}
                for arg_name in args_to_unpack & kwargs.keys():
                    arg_val = kwargs[arg_name]
                    if isinstance(arg_val, str):
                        unpacked_args[arg_name] = getattr(self, arg_val)[i]

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
        row_heigths = [max(plotter["height"] for plotter in plotters_row) for plotters_row in plotters]
        fig = plt.figure(figsize=(fig_width, sum(row_heigths)), constrained_layout=True)
        gridspecs = fig.add_gridspec(len(plotters), 1, height_ratios=row_heigths)

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
        plt.show()
        return self
