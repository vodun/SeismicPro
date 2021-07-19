"""Implements SeismicBatch class for processing a small subset of seismic gathers"""

import numpy as np

from .gather import Gather
from .semblance import Semblance, ResidualSemblance
from .decorators import create_batch_methods, apply_to_each_component
from .utils import to_list
from ..batchflow import Batch, action, DatasetIndex, BA


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
    method you want to apply:
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
        """Transform data for training model.

        The method produces two-stage data processing:
        1. Using the `mode` parameter, choose whether to `concatenate` or `stack` the input data along the specified
        `axis`.
        2. Expand dims of the resulted array along the `expand_dims_axis` axis.

        # переписать, говоря сначала о том, что может приходить, затем как это передается и обрабатывается
        There are two approaches to how the data from `src` can be stored and passed to this method:
        1. `src` as a string, is perceived as a name of the batch component with array-like of np.ndarrays - data
        that needs to be processed.
        2. `src` as an array-like of np.ndarrays, is perceived as a data to be stacked. There are many approaches to
        put the data directly to the `src`, most common is to use `BA` named expression. If a batch component contains
        array with insatnces of some classes, `BA` allows to extract the data from a certain attribute of the class and
        pass it directly to the `src` parameter.

        Examples
        --------
        Load traces. Then exctract items from `Gather` using `BA` named expression, concat them into an array with one
        dummy axis and save the result to the `inputs` component:
        >>> pipeline = (
                dataset.pipeline
                       .load(src='raw')
                       .make_model_inputs(src=BA('raw').data, dst='inputs', mode='c', axis=0, expand_dims_axis=1)
            )
        >>> batch = pipeline.next_batch(3)
        >>> batch.inputs.shape
        (3, 1, 1500)

        Parameters
        ----------
        src : src or np.ndarray
            A component's name or a array-like of np.ndarrays that needs to be processed.
        dst : src
            A component's name to store the result in.
        mode : {'c' or 's'}, optional, default to 'c'
            A mode that determines how to join a sequence of arrays:
            - 'c' apply `np.concatenate` to the data from `src` by specified `axis`.
            - 's' apply `np.stack` to the data from `src` by specified `axis`.
        axis : int or None, optional, default to 0
            An axis along which the arrays will be joined or stacked. There are two outcomes when the `axis` is None:
            - if `mode` is `c`, arrays are flattened before use.
            - if `mode` is `s`, the `axis` cannot be None.
            The maximal `axis` value equal to `data.ndim` - 1.
        expand_dims_axis : int or None, optional, default to None
            Insert a new axis at the `expand_dims_axis` position in the expanded array shape. The maximal axis value
            equal to `data.ndim` - 1. If `None`, the expansion does not occur.

        Returns
        -------
        self : Batch
            Batch with resulted `np.ndarray` in the `dst` component.

        Raises
        ------
        ValueError
            If given `mode` does not exist.
        """
        data = getattr(self, src) if isinstance(src, str) else src
        func = {'c': np.concatenate, 's': np.stack}.get(mode)
        if func is None:
            raise ValueError(f"Unknown mode '{mode}', must be 'c' or 's'")
        data = func(data, axis=axis)

        if expand_dims_axis is not None:
            data = np.expand_dims(data, axis=expand_dims_axis)
        setattr(self, dst, data)
        return self

    @action
    def split_model_outputs(self, src, dst, shapes):
        """Split data into multiple sub-arrays with shapes described in `shape`.

        A neural network model has a stacked data as an input and returns a stacked predictions, that needs to be
        split to the batches with equals shapes as before stack. This method is split the data using an array named
        `shape` that contains size of every batch.

        Examples
        --------
        Load traces, use model to predict a mask, split results and save it to the `output` batch component:
        >>> pipeline = (
                dataset.pipeline
                       .load(src='raw')
                       .init_variable('preds')
                       .init_model(mode='dynamic', model_class=UNet, name='model', config=config)
                       .make_model_inputs(src=BA('raw').data, dst='inputs', mode='c', axis=0, expand_dims_axis=1)
                       .predict_model('model', B('inputs'), fetches='predictions', save_to=B('preds'))
                       .split_model_outputs(src='preds', dst='outputs', shapes=BA('raw').shape[:, 0])
            )
        >>> batch = pipeline.next_batch(3)
        >>> len(batch.outputs)
        3
        >>> batch.outputs[0].shape
        (1, 1, 1500)

        Parameters
        ----------
        src : str
            A component's name with data to split.
        dst : str or BA
            - if `stc`, save resulted sub-arrays into a batch component named `dst`.
            - if `BA`, save resulted sub-arrays into a class attribute described in `BA` named expression.
        shapes : 1d array-like
            An array with sizes of every sub-array. The length of this array must be equal to the current batch size.
            If the size of the last group is less than the number of remaining elements in `src`, these elements will
            still be written to this group. For example, ``[2, 3, 2]`` would result in

                - src[:2]
                - src[2:5]
                - src[5:]

        Returns
        -------
        self : Batch
            Batch with split data.
        """
        data = getattr(self, src)
        shapes = np.cumsum(shapes)[:-1]
        splitted_data = np.split(data, shapes)

        if isinstance(dst, str):
            setattr(self, dst, splitted_data)
        elif isinstance(dst, BA):
            dst.set(value=splitted_data)
        else:
            ValueError(f'dst must be `str` or `BA named expression`, not {type(dst)}.')
        return self
