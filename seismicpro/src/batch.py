"""Implements SeismicBatch class for processing a small subset of seismic gathers"""

import numpy as np

from .gather import Gather
from .semblance import Semblance, ResidualSemblance
from .decorators import create_batch_methods, apply_to_each_component
from .utils import to_list
from ..batchflow import Batch, action, DatasetIndex, NamedExpression


@create_batch_methods(Gather, Semblance, ResidualSemblance)
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
    @property
    def nested_indices(self):
        """list: indices of the batch each additionally wrapped into a list. If used as an `init` function in
        `inbatch_parallel` decorator, each index will be passed to its parallelly executed callable as a tuple, not
        individual level values."""
        if isinstance(self.indices, np.ndarray):
            return self.indices.tolist()
        return [[index] for index in self.indices]

    def _init_component(self, *args, dst, **kwargs):
        """Create and preallocate new attributes with names listed in `dst` if they don't exist and return
        `self.nested_indices`. This method is typically used as a default `init` function in `inbatch_parallel`
        decorator."""
        _ = args, kwargs
        dst = to_list(dst)
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
        ...     .make_model_inputs(src=BA('survey').data, dst='inputs', mode='c', axis=0, expand_dims_axis=1)
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
        ...     .make_model_inputs(src=BA('survey').data, dst='inputs', mode='c', axis=0, expand_dims_axis=1)
        ...     .predict_model('model', B('inputs'), fetches='predictions', save_to=B('predictions'))
        ...     .split_model_outputs(src='predictions', dst='outputs', shapes=BA('survey').shape[:, 0])
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
