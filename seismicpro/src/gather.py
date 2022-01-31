"""Implements Gather class that represents a group of seismic traces that share some common acquisition parameter"""

import os
import warnings
from copy import deepcopy
from textwrap import dedent

import segyio
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .cropped_gather import CroppedGather
from .muting import Muter
from .semblance import Semblance, ResidualSemblance
from .velocity_cube import StackingVelocity, VelocityCube
from .decorators import batch_method, plotter
from .utils import normalization, correction
from .utils import (to_list, convert_times_to_mask, convert_mask_to_pick, times_to_indices, mute_gather, make_origins,
                    set_ticks, set_text_formatting)

class Gather:
    """A class representing a single seismic gather.

    A gather is a collection of seismic traces that share some common acquisition parameter (same index value of the
    generating survey header in our case). Unlike `Survey`, `Gather` instance stores loaded seismic traces along with
    a corresponding subset of its parent survey header.

    `Gather` instance can be created in three main ways:
    1. Either by calling `Survey.sample_gather` to get a randomly selected gather,
    2. Or by calling `Survey.get_gather` to get a particular gather by its index value,
    3. Or by calling `Index.get_gather` to get a particular gather by its index value from a specified survey.

    Most of the methods change gather data inplace, thus `Gather.copy` may come in handy to keep the original gather
    available.

    Examples
    --------
    Let's load a randomly selected common source gather, sort it by offset and plot:
    >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
    >>> gather = survey.sample_gather().sort(by="offset")
    >>> gather.plot()

    Parameters
    ----------
    headers : pd.DataFrame
        A subset of parent survey header with common index value defining the gather.
    data : 2d np.ndarray
        Trace data of the gather with (num_traces, trace_lenght) layout.
    samples : 1d np.ndarray of floats
        Recording time for each trace value. Measured in milliseconds.
    sample_rate : float
        Sample rate of seismic traces. Measured in milliseconds.
    survey : Survey
        A survey that generated the gather.

    Attributes
    ----------
    headers : pd.DataFrame
        A subset of parent survey header with common index value defining the gather.
    data : 2d np.ndarray
        Trace data of the gather with (num_traces, trace_lenght) layout.
    samples : 1d np.ndarray of floats
        Recording time for each trace value. Measured in milliseconds.
    sample_rate : float
        Sample rate of seismic traces. Measured in milliseconds.
    survey : Survey
        A survey that generated the gather.
    sort_by : None or str
        Headers column that was used for gather sorting. If `None`, no sorting was performed.
    """
    def __init__(self, headers, data, samples, sample_rate, survey):
        self.headers = headers
        self.data = data
        self.samples = samples
        self.sample_rate = sample_rate
        self.survey = survey
        self.sort_by = None

    @property
    def times(self):
        """1d np.ndarray of floats: Recording time for each trace value. Measured in milliseconds."""
        return self.samples

    @property
    def offsets(self):
        """1d np.ndarray of floats: The distance between source and receiver for each trace. Measured in meters."""
        return self.headers['offset'].values

    @property
    def shape(self):
        """tuple with 2 elements: The number of traces in the gather and trace length in samples."""
        return self.data.shape

    @property
    def n_traces(self):
        """int: The number of traces in the gather."""
        return self.shape[0]

    @property
    def n_samples(self):
        """int: Trace length in samples."""
        return self.shape[1]

    def __getitem__(self, key):
        """Either select gather headers values by their names or create a new `Gather` with specified traces and
        samples depending on the key type.

        Notes
        -----
        1. If the data after `__getitem__` is no longer sorted, `sort_by` attribute in the resulting `Gather` will be
        set to `None`.
        2. If headers selection is performed, a 2d array is always returned even for a single header.

        Parameters
        ----------
        key : str, list of str, int, list, tuple, slice
            If str or list of str, gather headers to get as a 2d np.ndarray.
            Otherwise, indices of traces and samples to get. In this case, __getitem__ behaviour almost coincides with
            np.ndarray indexing and slicing except for cases, when resulting ndim is not preserved or joint indexation
            of gather attributes becomes ambiguous (e.g. gather[[0, 1], [0, 1]]).

        Returns
        -------
        result : 2d np.ndarray or Gather
            Headers values or Gather with a specified subset of traces and samples.

        Raises
        ------
        ValueError
            If the resulting gather is empty, or data ndim has changed, or joint attribute indexation is ambiguous.
        """
        # If key is str or array of str, treat it as names of headers columns
        keys_array = np.array(to_list(key))
        if keys_array.dtype.type == np.str_:
            self.validate(required_header_cols=keys_array)
            # Avoid using direct pandas indexing to speed up multiple headers selection from gathers with a small
            # number of traces
            headers = []
            for col in keys_array:
                header = self.headers[col] if col in self.headers.columns else self.headers.index.get_level_values(col)
                headers.append(header.values)
            return np.column_stack(headers)

        # Perform traces and samples selection
        key = (key, ) if not isinstance(key, tuple) else key
        key = key + (slice(None), ) if len(key) == 1 else key
        indices = ()
        for axis_indexer, axis_shape in zip(key, self.shape):
            if isinstance(axis_indexer, (int, np.integer)):
                # Convert negative array index to a corresponding positive one
                axis_indexer %= axis_shape
                # Switch from simple indexing to a slice to keep array dims
                axis_indexer = slice(axis_indexer, axis_indexer+1)
            elif isinstance(axis_indexer, tuple):
                # Force advanced indexing for `samples`
                axis_indexer = list(axis_indexer)
            indices = indices + (axis_indexer, )

        new_self = self.copy(ignore=['data', 'headers', 'samples'])
        new_self.data = self.data[indices]
        if new_self.data.ndim != 2:
            raise ValueError("Data ndim is not preserved or joint indexation of gather attributes becomes ambiguous "
                             "after indexation")
        if new_self.data.size == 0:
            raise ValueError("Empty gather after indexation")

        # The two-dimensional `indices` array describes the indices of the traces and samples to be obtained,
        # respectively.
        new_self.headers = self.headers.iloc[indices[0]]
        new_self.samples = self.samples[indices[1]]

        # Check that `sort_by` still represents the actual trace sorting as it might be changed during getitem.
        if new_self.sort_by is not None and not new_self.headers[new_self.sort_by].is_monotonic_increasing:
            new_self.sort_by = None
        return new_self

    def __setitem__(self, key, value):
        """Set given values to selected gather headers.

        Parameters
        ----------
        key : str or list of str
            Gather headers to set values for.
        value : np.ndarray
            Headers values to set.
        """
        key = to_list(key)
        val = pd.DataFrame(value, columns=key, index=self.headers.index)
        self.headers[key] = val

    def __str__(self):
        """Print gather metadata including information about its survey, headers and traces."""

        # Calculate offset range
        offsets = self.headers.get('offset')
        offset_range = f'[{np.min(offsets)} m, {np.max(offsets)} m]' if offsets is not None else None

        # Determine index value
        index = np.unique(self.headers.index)
        index = 'combined' if len(index) > 1 else index.item()

        # Count the number of zero/constant traces
        n_dead_traces = np.isclose(np.max(self.data, axis=1), np.min(self.data, axis=1)).sum()
        msg = f"""
        Parent survey path:          {self.survey.path}
        Parent survey name:          {self.survey.name}

        Number of traces:            {self.n_traces}
        Trace length:                {self.n_samples} samples
        Sample rate:                 {self.sample_rate} ms
        Times range:                 [{min(self.samples)} ms, {max(self.samples)} ms]
        Offsets range:               {offset_range}

        Index name(s):               {', '.join(self.headers.index.names)}
        Index value:                 {index}
        Gather sorting:              {self.sort_by}

        Gather statistics:
        Number of dead traces:       {n_dead_traces}
        mean | std:                  {np.mean(self.data):>10.2f} | {np.std(self.data):<10.2f}
         min | max:                  {np.min(self.data):>10.2f} | {np.max(self.data):<10.2f}
         q01 | q99:                  {self.get_quantile(0.01):>10.2f} | {self.get_quantile(0.99):<10.2f}
        """
        return dedent(msg)

    def info(self):
        """Print gather metadata including information about its survey, headers and traces."""
        print(self)

    def get_coords(self, coords_columns="index"):
        """Get spatial coordinates of the gather.

        Parameters
        ----------
        coords_columns : None, "index" or 2 element array-like, defaults to "index"
            - If `None`, (`None`, `None`) tuple is returned.
            - If "index", unique index value is used to define gather coordinates
            - If 2 element array-like, `coords_columns` define gather headers to get x and y coordinates from.
            In the last two cases index or column values are supposed to be unique for all traces in the gather.

        Returns
        -------
        coords : tuple with 2 elements
            Gather spatial coordinates.

        Raises
        ------
        ValueError
            If gather coordinates are non-unique or more than 2 columns were passed.
        """
        if coords_columns is None:
            return (None, None)
        if coords_columns == "index":
            coords_columns = list(self.headers.index.names)
        coords = np.unique(self.headers.reset_index()[coords_columns].values, axis=0)
        if coords.shape[0] != 1:
            raise ValueError("Gather coordinates are non-unique")
        if coords.shape[1] != 2:
            raise ValueError(f"Gather position must be defined by exactly two coordinates, not {coords.shape[1]}")
        return tuple(coords[0].tolist())

    @batch_method(target='threads', copy_src=False)
    def copy(self, ignore=None):
        """Perform a deepcopy of all gather attributes except for `survey` and those specified in ignore, which are
        kept unchanged.

        Parameters
        ----------
        ignore : str or array of str, defaults to None
            Attributes that won't be copied.

        Returns
        -------
        copy : Gather
            Copy of the gather.
        """
        ignore_attrs = set() if ignore is None else set(to_list(ignore))
        ignore_attrs = [getattr(self, attr) for attr in ignore_attrs | {'survey'}]

        # Construct a memo dict with attributes, that should not be copied
        memo = {id(attr): attr for attr in ignore_attrs}
        return deepcopy(self, memo)

    @batch_method(target='for')
    def get_item(self, *args):
        """An interface for `self.__getitem__` method."""
        return self[args if len(args) > 1 else args[0]]

    def _validate_header_cols(self, required_header_cols):
        """Check if the gather headers contain all columns from `required_header_cols`."""
        header_cols = set(self.headers.columns) | set(self.headers.index.names)
        missing_headers = set(to_list(required_header_cols)) - header_cols
        if missing_headers:
            err_msg = "The following headers must be preloaded: {}"
            raise ValueError(err_msg.format(", ".join(missing_headers)))

    def _validate_sorting(self, required_sorting):
        """Check if the gather is sorted by `required_sorting` header."""
        if self.sort_by != required_sorting:
            raise ValueError(f"Gather should be sorted by {required_sorting} not {self.sort_by}")

    def validate(self, required_header_cols=None, required_sorting=None):
        """Perform the following checks for a gather:
            1. Its header contains all columns from `required_header_cols`,
            2. It is sorted by `required_sorting` header.

        Parameters
        ----------
        required_header_cols : None or str or array-like of str, defaults to None
            Required gather header columns. If `None`, no check is performed.
        required_sorting : None or str, defaults to None
            Required gather sorting. If `None`, no check is performed.

        Returns
        -------
        self : Gather
            Self unchanged.

        Raises
        ------
        ValueError
            If any of checks above failed.
        """
        if required_header_cols is not None:
            self._validate_header_cols(required_header_cols)
        if required_sorting is not None:
            self._validate_sorting(required_sorting)
        return self

    #------------------------------------------------------------------------#
    #                              Dump methods                              #
    #------------------------------------------------------------------------#

    @batch_method(target='for', force=True, copy_src=False)
    def dump(self, path, name=None, copy_header=False):
        """Save the gather to a `.sgy` file.

        Notes
        -----
        All binary and textual headers are copied from the parent SEG-Y file unchanged.

        Parameters
        ----------
        path : str
            The directory to dump the gather in.
        name : str, optional, defaults to None
            The name of the file. If `None`, the concatenation of the survey name and the value of gather index will
            be used.
        copy_header : bool, optional, defaults to False
            Whether to copy the headers that weren't loaded during Survey creation from the parent SEG-Y file.

        Returns
        -------
        self : Gather
            Gather unchanged.

        Raises
        ------
        ValueError
            If empty `name` was specified.
        """
        parent_handler = self.survey.segy_handler

        if name is None:
            name = "_".join(map(str, [self.survey.name] + to_list(self.headers.index.values[0])))
        if name == "":
            raise ValueError("Argument `name` can not be empty.")
        if not os.path.splitext(name)[1]:
            name += ".sgy"
        full_path = os.path.join(path, name)

        os.makedirs(path, exist_ok=True)
        # Create segyio spec. We choose only specs that relate to unstructured data.
        spec = segyio.spec()
        spec.samples = self.samples
        spec.ext_headers = parent_handler.ext_headers
        spec.format = parent_handler.format
        spec.tracecount = self.n_traces

        trace_headers = self.headers.reset_index()

        # Remember ordinal numbers of traces in the parent SEG-Y file to further copy their headers
        # and reset them to start from 1 in the resulting file to match SEG-Y standard.
        trace_ids = trace_headers["TRACE_SEQUENCE_FILE"].values - 1
        trace_headers["TRACE_SEQUENCE_FILE"] = np.arange(len(trace_headers)) + 1

        # Keep only headers, defined by SEG-Y standard.
        used_header_names = set(trace_headers.columns) & set(segyio.tracefield.keys.keys())
        trace_headers = trace_headers[used_header_names]

        # Now we change column name's into byte number based on the SEG-Y standard.
        trace_headers.rename(columns=lambda col_name: segyio.tracefield.keys[col_name], inplace=True)
        trace_headers_dict = trace_headers.to_dict("index")

        with segyio.create(full_path, spec) as dump_handler:
            # Copy binary headers from the parent SEG-Y file. This is possibly incorrect and needs to be checked
            # if the number of traces or sample ratio changes.
            # TODO: Check if bin headers matter
            dump_handler.bin = parent_handler.bin

            # Copy textual headers from the parent SEG-Y file.
            for i in range(spec.ext_headers + 1):
                dump_handler.text[i] = parent_handler.text[i]

            # Dump traces and their headers. Optionally copy headers from the parent SEG-Y file.
            dump_handler.trace = self.data
            for i, dump_h in trace_headers_dict.items():
                if copy_header:
                    dump_handler.header[i].update(parent_handler.header[trace_ids[i]])
                dump_handler.header[i].update(dump_h)
        return self

    #------------------------------------------------------------------------#
    #                         Normalization methods                          #
    #------------------------------------------------------------------------#

    def _apply_agg_func(self, func, tracewise, **kwargs):
        """Apply a `func` either to entire gather's data or to each trace independently.

        Notes
        -----
        `func` must accept an `axis` argument.

        Parameters
        ----------
        func : callable
            Function to be applied to the gather's data.
        tracewise : bool
            If `True`, the `func` is applied to each trace independently, otherwise to the entire gather's data.
        kwargs : misc, optional
            Additional keyword arguments to `func`.

        Returns
        -------
        result : misc
            The result of the application of the `func` to the gather's data.
        """
        axis = 1 if tracewise else None
        return func(self.data, axis=axis, **kwargs)

    def get_quantile(self, q, tracewise=False, use_global=False):
        """Calculate the `q`-th quantile of the gather or fetch the global quantile from the parent survey.

        Notes
        -----
        The `tracewise` mode is only available when `use_global` is `False`.

        Parameters
        ----------
        q : float or array-like of floats
            Quantile or a sequence of quantiles to compute, which must be between 0 and 1 inclusive.
        tracewise : bool, optional, default False
            If `True`, the quantiles are computed for each trace independently, otherwise for the entire gather.
        use_global : bool, optional, default False
            If `True`, the survey's quantiles are used, otherwise the quantiles are computed for the gather data.

        Returns
        -------
        q : float or array-like of floats
            The `q`-th quantile values.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        if use_global:
            return self.survey.get_quantile(q)
        quantiles = self._apply_agg_func(func=np.nanquantile, tracewise=tracewise, q=q).astype(np.float32)
        # return the same type as q in case of global calculation: either single float or array-like
        return quantiles.item() if not tracewise and quantiles.ndim == 0 else quantiles

    @batch_method(target='threads')
    def scale_standard(self, tracewise=True, use_global=False, eps=1e-10):
        r"""Standardize the gather by removing the mean and scaling to unit variance.

        The standard score of a gather `g` is calculated as:
        :math:`G = \frac{g - m}{s + eps}`,
        where:
        `m` - the mean of the gather or global average if `use_global=True`,
        `s` - the standard deviation of the gather or global standard deviation if `use_global=True`,
        `eps` - a constant that is added to the denominator to avoid division by zero.

        Notes
        -----
        1. The presence of NaN values in the gather will lead to incorrect behavior of the scaler.
        2. Standardization is performed inplace.

        Parameters
        ----------
        tracewise : bool, optional, defaults to True
            If `True`, mean and standard deviation are calculated for each trace independently. Otherwise they are
            calculated for the entire gather.
        use_global : bool, optional, defaults to False
            If `True`, parent survey's mean and std are used, otherwise gather statistics are computed.
        eps : float, optional, defaults to 1e-10
            A constant to be added to the denominator to avoid division by zero.

        Returns
        -------
        self : Gather
            Standardized gather.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        if use_global:
            if not self.survey.has_stats:
                raise ValueError('Global statistics were not calculated, call `Survey.collect_stats` first.')
            mean = self.survey.mean
            std = self.survey.std
        else:
            mean = self._apply_agg_func(func=np.mean, tracewise=tracewise, keepdims=True)
            std = self._apply_agg_func(func=np.std, tracewise=tracewise, keepdims=True)
        self.data = normalization.scale_standard(self.data, mean, std, np.float32(eps))
        return self

    @batch_method(target='threads')
    def scale_maxabs(self, q_min=0, q_max=1, tracewise=True, use_global=False, clip=False, eps=1e-10):
        r"""Scale the gather by its maximum absolute value.

        Maxabs scale of the gather `g` is calculated as:
        :math: `G = \frac{g}{m + eps}`,
        where:
        `m` - the maximum of absolute values of `q_min`-th and `q_max`-th quantiles,
        `eps` - a constant that is added to the denominator to avoid division by zero.

        Quantiles are used to minimize the effect of amplitude outliers on the scaling result. Default 0 and 1
        quantiles represent the minimum and maximum values of the gather respectively and result in usual max-abs
        scaler behavior.

        Notes
        -----
        1. The presence of NaN values in the gather will lead to incorrect behavior of the scaler.
        2. Maxabs scaling is performed inplace.

        Parameters
        ----------
        q_min : float, optional, defaults to 0
            A quantile to be used as a gather minimum during scaling.
        q_max : float, optional, defaults to 1
            A quantile to be used as a gather maximum during scaling.
        tracewise : bool, optional, defaults to True
            If `True`, quantiles are calculated for each trace independently. Otherwise they are calculated for the
            entire gather.
        use_global : bool, optional, defaults to False
            If `True`, parent survey's quantiles are used, otherwise gather quantiles are computed.
        clip : bool, optional, defaults to False
            Whether to clip the scaled gather to the [-1, 1] range.
        eps : float, optional, defaults to 1e-10
            A constant to be added to the denominator to avoid division by zero.

        Returns
        -------
        self : Gather
            Scaled gather.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        min_value, max_value = self.get_quantile([q_min, q_max], tracewise=tracewise, use_global=use_global)
        self.data = normalization.scale_maxabs(self.data, min_value, max_value, clip, np.float32(eps))
        return self

    @batch_method(target='threads')
    def scale_minmax(self, q_min=0, q_max=1, tracewise=True, use_global=False, clip=False, eps=1e-10):
        r"""Linearly scale the gather to a [0, 1] range.

        The transformation of the gather `g` is given by:
        :math:`G=\frac{g - min}{max - min + eps}`
        where:
        `min` and `max` - `q_min`-th and `q_max`-th quantiles respectively,
        `eps` - a constant that is added to the denominator to avoid division by zero.

        Notes
        -----
        1. The presence of NaN values in the gather will lead to incorrect behavior of the scaler.
        2. Minmax scaling is performed inplace.

        Parameters
        ----------
        q_min : float, optional, defaults to 0
            A quantile to be used as a gather minimum during scaling.
        q_max : float, optional, defaults to 1
            A quantile to be used as a gather maximum during scaling.
        tracewise : bool, optional, defaults to True
            If `True`, quantiles are calculated for each trace independently. Otherwise they are calculated for the
            entire gather.
        use_global : bool, optional, defaults to False
            If `True`, parent survey's quantiles are used, otherwise gather quantiles are computed.
        clip : bool, optional, defaults to False
            Whether to clip the scaled gather to the [0, 1] range.
        eps : float, optional, defaults to 1e-10
            A constant to be added to the denominator to avoid division by zero.

        Returns
        -------
        self : Gather
            Scaled gather.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        min_value, max_value = self.get_quantile([q_min, q_max], tracewise=tracewise, use_global=use_global)
        self.data = normalization.scale_minmax(self.data, min_value, max_value, clip, np.float32(eps))
        return self

    #------------------------------------------------------------------------#
    #                    First-breaks processing methods                     #
    #------------------------------------------------------------------------#

    @batch_method(target="threads", copy_src=False)
    def pick_to_mask(self, first_breaks_col="FirstBreak"):
        """Convert first break times to a binary mask with the same shape as `gather.data` containing zeros before the
        first arrivals and ones after for each trace.

        Parameters
        ----------
        first_breaks_col : str, optional, defaults to 'FirstBreak'
            A column of `self.headers` that contains first arrival times, measured in milliseconds.

        Returns
        -------
        gather : Gather
            A new `Gather` with calculated first breaks mask in its `data` attribute.
        """
        mask = convert_times_to_mask(times=self[first_breaks_col].ravel(), sample_rate=self.sample_rate,
                                     mask_length=self.shape[1]).astype(np.int32)
        gather = self.copy(ignore='data')
        gather.data = mask
        return gather


    @batch_method(target='for', args_to_unpack='save_to')
    def mask_to_pick(self, threshold=0.5, first_breaks_col="FirstBreak", save_to=None):
        """Convert a first break mask saved in `data` into times of first arrivals.

        For a given trace each value of the mask represents the probability that the corresponding index is greater
        than the index of the first break.

        Notes
        -----
        A detailed description of conversion heuristic used can be found in :func:`~general_utils.convert_mask_to_pick`
        docs.

        Parameters
        ----------
        threshold : float, optional, defaults to 0.5
            A threshold for trace mask value to refer its index to be either pre- or post-first break.
        first_breaks_col : str, optional, defaults to 'FirstBreak'
            Headers column to save first break times to.
        save_to : Gather, optional, defaults to None
            An extra `Gather` to save first break times to. Generally used to conveniently pass first break times from
            a `Gather` instance with a first break mask to an original `Gather`.

        Returns
        -------
        self : Gather
            A gather with first break times in headers column defined by `first_breaks_col`.
        """
        picking_times = convert_mask_to_pick(self.data, self.sample_rate, threshold)
        self[first_breaks_col] = picking_times
        if save_to is not None:
            save_to[first_breaks_col] = picking_times
        return self

    @batch_method(target='for', use_lock=True)
    def dump_first_breaks(self, path, trace_id_cols=('FieldRecord', 'TraceNumber'), first_breaks_col='FirstBreak',
                          col_space=8, encoding="UTF-8"):
        """ Save first break picking times to a file.

        Each line in the resulting file corresponds to one trace, where all columns but
        the last one store values from `trace_id_cols` headers and identify the trace
        while the last column stores first break time from `first_breaks_col` header.

        Parameters
        ----------
        path : str
            Path to the file.
        trace_id_cols : tuple of str, defaults to ('FieldRecord', 'TraceNumber')
            Columns names from `self.headers` that act as trace id. These would be present in the file.
        first_breaks_col : str, defaults to 'FirstBreak'
            Column name from `self.headers` where first break times are stored.
        col_space : int, defaults to 8
            The minimum width of each column.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.

        Returns
        -------
        self : Gather
            Gather unchanged
        """
        rows = self[to_list(trace_id_cols) + [first_breaks_col]]

        # SEG-Y specification states that all headers values are integers, but first break values can be float
        row_fmt = '{:{col_space}.0f}' * (rows.shape[1] - 1) + '{:{col_space}.2f}\n'
        fmt = row_fmt * len(rows)
        rows_as_str = fmt.format(*rows.ravel(), col_space=col_space)

        with open(path, 'a', encoding=encoding) as f:
            f.write(rows_as_str)
        return self

    #------------------------------------------------------------------------#
    #                         Gather muting methods                          #
    #------------------------------------------------------------------------#

    @batch_method(target="for", copy_src=False)
    def create_muter(self, mode="first_breaks", **kwargs):
        """Create an instance of :class:`~.Muter` class.

        This method redirects the call into a corresponding `Muter.from_{mode}` classmethod. The created object is
        callable and returns times up to which muting should be performed for given offsets. A detailed description of
        `Muter` instance can be found in :class:`~muting.Muter` docs.

        Parameters
        ----------
        mode : {"points", "file", "first_breaks"}, optional, defaults to "first_breaks"
            Type of `Muter` to create.
        kwargs : misc, optional
            Additional keyword arguments to `Muter.from_{mode}`.

        Returns
        -------
        muter : Muter
            Created muter.

        Raises
        ------
        ValueError
            If given `mode` does not exist.
        """
        builder = getattr(Muter, f"from_{mode}", None)
        if builder is None:
            raise ValueError(f"Unknown mode {mode}")

        if mode == "first_breaks":
            first_breaks_col = kwargs.pop("first_breaks_col", "FirstBreak")
            return builder(offsets=self.offsets, times=self[first_breaks_col], **kwargs)
        return builder(**kwargs)

    @batch_method(target="threads", args_to_unpack="muter")
    def mute(self, muter, fill_value=0):
        """Mute the gather using given `muter`.

        The muting operation is performed by setting gather values above an offset-time boundary defined by `muter` to
        `fill_value`.

        Parameters
        ----------
        muter : Muter
            An object that defines muting times by gather offsets.
        fill_value : float, defaults to 0
            A value to fill the muted part of the gather with.

        Returns
        -------
        self : Gather
            Muted gather.
        """
        self.data = mute_gather(gather_data=self.data, muting_times=muter(self.offsets),
                                sample_rate=self.sample_rate, fill_value=fill_value)
        return self

    #------------------------------------------------------------------------#
    #                     Semblance calculation methods                      #
    #------------------------------------------------------------------------#

    @batch_method(target="threads", copy_src=False)
    def calculate_semblance(self, velocities, win_size=25):
        """Calculate vertical velocity semblance for the gather.

        Notes
        -----
        The gather should be sorted by offset. A detailed description of vertical velocity semblance and its
        computation algorithm can be found in :func:`~semblance.Semblance` docs.

        Examples
        --------
        Calculate semblance for 200 velocities from 2000 to 6000 m/s and a temporal window size of 8 samples:
        >>> gather = gather.sort(by="offset")
        >>> semblance = gather.calculate_semblance(velocities=np.linspace(2000, 6000, 200), win_size=8)

        Parameters
        ----------
        velocities : 1d np.ndarray
            Range of velocity values for which semblance is calculated. Measured in meters/seconds.
        win_size : int, optional, defaults to 25
            Temporal window size used for semblance calculation. The higher the `win_size` is, the smoother the
            resulting semblance will be but to the detriment of small details. Measured in samples.

        Returns
        -------
        semblance : Semblance
            Calculated vertical velocity semblance.

        Raises
        ------
        ValueError
            If the gather is not sorted by offset.
        """
        self.validate(required_sorting="offset")
        return Semblance(gather=self, velocities=velocities, win_size=win_size)

    @batch_method(target="threads", args_to_unpack="stacking_velocity", copy_src=False)
    def calculate_residual_semblance(self, stacking_velocity, n_velocities=140, win_size=25, relative_margin=0.2):
        """Calculate residual vertical velocity semblance for the gather and a chosen stacking velocity.

        Notes
        -----
        The gather should be sorted by offset. A detailed description of residual vertical velocity semblance and its
        computation algorithm can be found in :func:`~semblance.ResidualSemblance` docs.

        Examples
        --------
        Calculate residual semblance for a gather and a stacking velocity, loaded from a file:
        >>> gather = gather.sort(by="offset")
        >>> velocity = StackingVelocity.from_file(velocity_path)
        >>> residual = gather.calculate_residual_semblance(velocity, n_velocities=100, win_size=8)

        Parameters
        ----------
        stacking_velocity : StackingVelocity
            Stacking velocity around which residual semblance is calculated.
        n_velocities : int, optional, defaults to 140
            The number of velocities to compute residual semblance for.
        win_size : int, optional, defaults to 25
            Temporal window size used for semblance calculation. The higher the `win_size` is, the smoother the
            resulting semblance will be but to the detriment of small details. Measured in samples.
        relative_margin : float, optional, defaults to 0.2
            Relative velocity margin, that determines the velocity range for semblance calculation for each time `t` as
            `stacking_velocity(t)` * (1 +- `relative_margin`).

        Returns
        -------
        semblance : ResidualSemblance
            Calculated residual vertical velocity semblance.

        Raises
        ------
        ValueError
            If the gather is not sorted by offset.
        """
        self.validate(required_sorting="offset")
        return ResidualSemblance(gather=self, stacking_velocity=stacking_velocity, n_velocities=n_velocities,
                                 win_size=win_size, relative_margin=relative_margin)

    #------------------------------------------------------------------------#
    #                           Gather corrections                           #
    #------------------------------------------------------------------------#

    @batch_method(target="threads", args_to_unpack="stacking_velocity")
    def apply_nmo(self, stacking_velocity, coords_columns="index"):
        """Perform gather normal moveout correction using given stacking velocity.

        Notes
        -----
        A detailed description of NMO correction can be found in :func:`~correction.apply_nmo` docs.

        Parameters
        ----------
        stacking_velocity : StackingVelocity or VelocityCube
            Stacking velocities to perform NMO correction with. `StackingVelocity` instance is used directly. If
            `VelocityCube` instance is passed, a `StackingVelocity` corresponding to gather coordinates is fetched
            from it.
        coords_columns : None, "index" or 2 element array-like, defaults to "index"
            Header columns to get spatial coordinates of the gather from to fetch `StackingVelocity` from
            `VelocityCube`. See :func:`~Gather.get_coords` for more details.

        Returns
        -------
        self : Gather
            NMO corrected gather.

        Raises
        ------
        ValueError
            If `stacking_velocity` is not a `StackingVelocity` or `VelocityCube` instance.
        """
        if isinstance(stacking_velocity, VelocityCube):
            stacking_velocity = stacking_velocity(*self.get_coords(coords_columns), create_interpolator=False)
        if not isinstance(stacking_velocity, StackingVelocity):
            raise ValueError("Only VelocityCube or StackingVelocity instances can be passed as a stacking_velocity")
        velocities_ms = stacking_velocity(self.times) / 1000  # from m/s to m/ms
        self.data = correction.apply_nmo(self.data, self.times, self.offsets, velocities_ms, self.sample_rate)
        return self

    #------------------------------------------------------------------------#
    #                       General processing methods                       #
    #------------------------------------------------------------------------#

    @batch_method(target="for")
    def sort(self, by):
        """Sort gather `headers` and traces by specified header column.

        Parameters
        ----------
        by : str
            `headers` column name to sort the gather by.

        Returns
        -------
        self : Gather
            Gather sorted by `by` column. Sets `sort_by` attribute to `by`.

        Raises
        ------
        TypeError
            If `by` is not str.
        ValueError
            If `by` column was not loaded in `headers`.
        """
        if not isinstance(by, str):
            raise TypeError(f'`by` should be str, not {type(by)}')
        self.validate(required_header_cols=by)
        order = np.argsort(self.headers[by].values, kind='stable')
        self.sort_by = by
        self.data = self.data[order]
        self.headers = self.headers.iloc[order]
        return self

    @batch_method(target="for")
    def get_central_cdp(self):
        """Get a central CDP gather from a supergather.

        A supergather has `SUPERGATHER_INLINE_3D` and `SUPERGATHER_CROSSLINE_3D` headers columns, whose values are
        equal to values in `INLINE_3D` and `CROSSLINE_3D` only for traces from the central CDP gather. Read more about
        supergather generation in :func:`~Survey.generate_supergathers` docs.

        Returns
        -------
        self : Gather
            `self` with only traces from the central CDP gather kept. Updates `self.headers` and `self.data` inplace.

        Raises
        ------
        ValueError
            If any of the `INLINE_3D`, `CROSSLINE_3D`, `SUPERGATHER_INLINE_3D` or `SUPERGATHER_CROSSLINE_3D` columns
            are not in `headers`.
        """
        self.validate(required_header_cols=["INLINE_3D", "SUPERGATHER_INLINE_3D",
                                            "CROSSLINE_3D", "SUPERGATHER_CROSSLINE_3D"])
        headers = self.headers.reset_index()
        mask = ((headers["SUPERGATHER_INLINE_3D"] == headers["INLINE_3D"]) &
                (headers["SUPERGATHER_CROSSLINE_3D"] == headers["CROSSLINE_3D"])).values
        self.headers = self.headers.loc[mask]
        self.data = self.data[mask]
        return self

    @batch_method(target="for")
    def stack(self):
        """Stack a gather by calculating mean value of all non-nan amplitudes for each time over the offset axis.

        The gather after stacking contains only one trace. Its `headers` is indexed by `INLINE_3D` and `CROSSLINE_3D`
        and has a single `TRACE_SEQUENCE_FILE` header with a value of 1.

        Notes
        -----
        Only a CDP gather indexed by `INLINE_3D` and `CROSSLINE_3D` can be stacked.

        Raises
        ------
        ValueError
            If the gather is not indexed by `INLINE_3D` and `CROSSLINE_3D` or traces from multiple CDP gathers are
            being stacked
        """
        line_cols = ["INLINE_3D", "CROSSLINE_3D"]
        self.validate(required_header_cols=line_cols)
        headers = self.headers.reset_index()[line_cols].drop_duplicates()
        if len(headers) != 1:
            raise ValueError("Only a single CDP gather can be stacked")
        self.headers = headers.set_index(line_cols)
        self.headers["TRACE_SEQUENCE_FILE"] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.data = np.nanmean(self.data, axis=0, keepdims=True)
        self.data = np.nan_to_num(self.data)
        return self

    def crop(self, origins, crop_shape, n_crops=1, stride=None, pad_mode='constant', **kwargs):
        """"Crop gather data.

        Parameters
        ----------
        origins : list, tuple, np.ndarray or str
            Origins define top-left corners for each crop (the first trace and the first time sample respectively)
            or a rule used to calculate them. All array-like values are cast to an `np.ndarray` and treated as origins
            directly, except for a 2-element tuple of `int`, which will be treated as a single individual origin.
            If `str`, represents a mode to calculate origins. Two options are supported:
            - "random": calculate `n_crops` crops selected randomly using a uniform distribution over the gather data,
              so that no crop crosses gather boundaries,
            - "grid": calculate a deterministic uniform grid of origins, whose density is determined by `stride`.
        crop_shape : tuple with 2 elements
            Shape of the resulting crops.
        n_crops : int, optional, defaults to 1
            The number of generated crops if `origins` is "random".
        stride : tuple with 2 elements, optional, defaults to crop_shape
            Steps between two adjacent crops along both axes if `origins` is "grid". The lower the value is, the more
            dense the grid of crops will be. An extra origin will always be placed so that the corresponding crop will
            fit in the very end of an axis to guarantee complete data coverage with crops regardless of passed
            `crop_shape` and `stride`.
        pad_mode : str or callable, optional, defaults to 'constant'
            Padding mode used when a crop with given origin and shape crossed boundaries of gather data. Passed
            directly to `np.pad`, see https://numpy.org/doc/stable/reference/generated/numpy.pad.html for more
            details.
        kwargs : dict, optional
            Additional keyword arguments to `np.pad`.

        Returns
        -------
        crops : CroppedGather
            Calculated gather crops.
        """
        origins = make_origins(origins, self.shape, crop_shape, n_crops, stride)
        return CroppedGather(self, origins, crop_shape, pad_mode, **kwargs)

    #------------------------------------------------------------------------#
    #                         Visualization methods                          #
    #------------------------------------------------------------------------#

    @plotter(figsize=(10, 7))
    def plot(self, mode="seismogram", title=None, x_ticker=None, y_ticker=None, ax=None, **kwargs):
        """ TODO
        Plot gather traces.

        The traces can be displayed in a number of representations, depending on the `mode` provided. Currently, the
        following options are supported:
        - `seismogram`: a 2d grayscale image of seismic traces. This mode supports the following `kwargs`:
            * `colorbar`: whether to add a colorbar to the right of the gather plot (defaults to `False`). If `dict`,
              defines extra keyword arguments for `matplotlib.figure.Figure.colorbar`.
            * `qvmin`, `qvmax`: quantile range of amplitude values covered by the colormap (defaults to 0.1 and 0.9),
            * Any additional arguments for `matplotlib.pyplot.imshow`. Note, that `vmin` and `vmax` arguments take
              priority over `qvmin` and `qvmax` respectively.
        - `wiggle`: an amplitude vs time plot for each trace of the gather as an oscillating line around its mean
          amplitude. This mode supports the following `kwargs`:
            * `std`: amplitude scaling factor. Higher values result in higher plot oscillations (defaults to 0.5),
            * `color`: defines a color for each trace. If a single color is given, it is applied to all the traces
              (defaults to black).

        Trace headers, whose values are measured in milliseconds (e.g. first break times) may be displayed over the
        gather plot if passed as `event_headers`. If `top_header` is passed, an auxiliary scatter plot of values of
        this header will be shown on top of the gather plot.

        Ticks and their labels for both `x` and `y` axes can be controlled via `x_ticker` and `y_ticker` parameters
        respectively. In the most general form, each of them is a `dict` with the following most commonly used keys:
        - `label`: source to get tick labels from. Can be either any gather header name or "index" (ordinal numbers of
          traces) for `x` axis and "times" or "samples" for `y` axis.
        - `round_to`: the number of decimal places to round tick labels to (defaults to 0).
        - `rotation`: the rotation angle of tick labels in degrees (defaults to 0).
        - One of the following keys, defining the way to place ticks:
            * `num`: place a given number of evenly-spaced ticks,
            * `step_ticks`: place ticks with a given step between two adjacent ones in samples (e.g. place a tick for
              every hundredth value in labels),
            * `step_labels`: place ticks with a given step between two adjacent ones in the units of the corresponding
              labels (e.g. place a tick every 200ms for `y` axis or every 300m offset for `x` axis).
        A short argument form allows defining both tickers as a single `str`, which will be treated as the value for
        the `label` key. See :func:`~plot_utils.set_ticks` for more details on the ticker parameters.

        Parameters
        ----------
        mode : "seismogram" or "wiggle", optional, defaults to "seismogram"
            A type of the gather representation to display:
            - "seismogram": a 2d grayscale image of seismic traces,
            - "wiggle": an amplitude vs time plot for each trace of the gather.
        event_headers : str, array-like or dict, optional, defaults to None
            Headers, whose values will be displayed over the gather plot. Must be measured in milliseconds.
            If `dict`, allows controlling scatter plot options and handling outliers (header values falling out the `y`
            axis range). The following keys are supported:
            - `headers`: header names, can be either `str` or an array-like.
            - `process_outliers`: an approach for outliers processing. Available options are:
                * `clip`: clip outliers to fit the range of `y` axis,
                * `discard`: do not display outliers,
                * `none`: plot all the header values (default behavior).
            - Any additional arguments for `matplotlib.axes.Axes.scatter`.
            If any dictionary value is array-like, each its element will be associated with the corresponding header.
            Otherwise, the single value will be used for all the scatter plots.
        top_header : str, optional, defaults to None
            A header name to plot on top of the gather plot.
        title : str or dict, optional, defaults to None
            If `str`, a title of the plot.
            If `dict`, should contain keyword arguments to pass to `matplotlib.axes.Axes.set_title`. In this case, the
            title string is stored under the `label` key.
        x_ticker : str or dict, optional, defaults to None
            Source to get `x` tick labels from and additional parameters to control their formatting and layout.
            If `str`, either any gather header name to use its values as labels or "index" to use ordinal numbers of
            traces in the gather.
            If `dict`, the source is specified under the "labels" key and the rest keys define labels formatting and
            layout, see :func:`~plot_utils.set_ticks` for more details.
            If not given, but the gather is sorted, `self.sort_by` will be passed as `x_ticker`. Otherwise, "index"
            will be passed.
        y_ticker : "time", "samples" or dict, optional, defaults to "time"
            Source to get `y` tick labels from and additional parameters to control their formatting and layout.
            If "time", the labels are the times of gather samples in milliseconds, if "samples" - ordinal numbers of
            gather samples.
            If `dict`, stores either "time" or "samples" under the "labels" key and the rest keys define labels
            formatting and layout, see :func:`~plot_utils.set_ticks` for more details.
        ax : matplotlib.axes.Axes, optional, defaults to None
            An axis of the figure to plot on. If not given, it will be created automatically.
        figsize : tuple, optional, defaults to (10, 7)
            Size of the figure to create if `ax` is not given. Measured in inches.
        save_to : str or dict, optional, defaults to None
            If `str`, a path to save the figure to.
            If `dict`, should contain keyword arguments to pass to `matplotlib.pyplot.savefig`. In this case, the path
            is stored under the `fname` key.
            If `None`, the figure is not saved.
        kwargs : misc, optional
            Additional keyword arguments to the plotter depending on the `mode`.

        Returns
        -------
        self : Gather
            Gather unchanged.

        Raises
        ------
        ValueError
            If given `mode` is unknown.
            If `colorbar` is not `bool` or `dict`.
            If the number of `colors` doesn't match the number of traces.
            If `event_headers` argument has the wrong format or given outlier processing mode is unknown.
            If `x_ticker` or `y_ticker` has the wrong format.
        """
        # Cast text-related parameters to dicts and add text formatting parameters from kwargs to each of them
        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)

        # Plot the gather depending on the mode passed
        plotters_dict = {
            "seismogram": self._plot_seismogram,
            "wiggle": self._plot_wiggle,
            "hist": self._plot_histogram,
        }
        if mode not in plotters_dict:
            raise ValueError(f"Unknown mode {mode}")
        ax = plotters_dict[mode](ax, x_ticker=x_ticker, y_ticker=y_ticker, **kwargs)
        ax.set_title(**{'label': None, **title})
        return self

    def _plot_histogram(self, ax, bins=50, x_tick_src="amplitude", log=False, x_ticker=None, y_ticker=None, grid=False,
                        **kwargs):
        """ TODO """
        data = self.data if x_tick_src=="amplitude" else self[x_tick_src]
        counts, _, _ = ax.hist(data.ravel(), bins=bins, **kwargs)
        set_ticks(ax, "x", tick_labels=None, **{"label": x_tick_src, 'round_to': None, **x_ticker})
        set_ticks(ax, "y", tick_labels=np.arange(0, counts.max()+1), **{"label": "counts", **y_ticker})

        ax.grid(grid)
        if log:
            ax.set_yscale("log")
        return ax

    # pylint: disable=too-many-arguments
    def _plot_seismogram(self, ax, colorbar=False, qvmin=0.1, qvmax=0.9, x_ticker=None, y_ticker=None,
                         x_tick_src=None, y_tick_src='time', event_headers=None, top_header=None, **kwargs):
        """Plot the gather as a 2d grayscale image of seismic traces."""
        # Make the axis divisible to further plot colorbar and header subplot
        divider = make_axes_locatable(ax)

        vmin, vmax = self.get_quantile([qvmin, qvmax])
        kwargs = {"cmap": "gray", "aspect": "auto", "vmin": vmin, "vmax": vmax, **kwargs}
        img = ax.imshow(self.data.T, **kwargs)
        if not isinstance(colorbar, (bool, dict)):
            raise ValueError(f"colorbar must be bool or dict but {type(colorbar)} was passed")
        if colorbar is not False:
            colorbar = {} if colorbar is True else colorbar
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax.figure.colorbar(img, cax=cax, **colorbar)
        return self._finalize_plot(ax, divider, event_headers, top_header, x_ticker, y_ticker, x_tick_src, y_tick_src)

    def _plot_wiggle(self, ax, std=0.5, color="black", x_ticker=None, y_ticker=None, x_tick_src=None,
                     y_tick_src='time', event_headers=None, top_header=None, **kwargs):
        """Plot the gather as an amplitude vs time plot for each trace."""
        # Make the axis divisible to further plot colorbar and header subplot
        divider = make_axes_locatable(ax)

        color = to_list(color)
        if len(color) == 1:
            color = color * self.n_traces
        elif len(color) != self.n_traces:
            raise ValueError('The number of items in `color` must match the number of plotted traces')

        y_coords = np.arange(self.n_samples)
        traces = std * (self.data - self.data.mean(axis=1, keepdims=True)) / (np.std(self.data) + 1e-10)
        for i, (trace, col) in enumerate(zip(traces, color)):
            ax.plot(i + trace, y_coords, color=col, **kwargs)
            ax.fill_betweenx(y_coords, i, i + trace, where=(trace > 0), color=col, **kwargs)
        ax.invert_yaxis()

        # Wiggle plot requires custom data interval for correct tick setting
        x_ticker.update({"tick_range":(0, self.n_traces-1)})
        return self._finalize_plot(ax, divider, event_headers, top_header, x_ticker, y_ticker, x_tick_src, y_tick_src)

    def _finalize_plot(self, ax, divider, event_headers, top_header, x_ticker, y_ticker, x_tick_src, y_tick_src):
        # Add headers scatter plot if needed
        if event_headers is not None:
            self._plot_headers(ax, event_headers)

        # Add a top subplot for given header if needed and set plot title
        top_ax = ax
        if top_header is not None:
            top_ax = self._plot_top_subplot(ax=ax, divider=divider, header_values=self[top_header].ravel())

        # Set axis ticks
        x_tick_src = (self.sort_by if self.sort_by is not None else "index") if x_tick_src is None else x_tick_src
        self._set_ticks(ax, axis="x", tick_src=x_tick_src, ticker=x_ticker)
        self._set_ticks(ax, axis="y", tick_src=y_tick_src, ticker=y_ticker)

        return top_ax

    @staticmethod
    def _parse_headers_kwargs(headers_kwargs, headers_key):
        """Construct a `dict` of kwargs for each header defined in `headers_kwargs` under `headers_key` key so that it
        contains all other keys from `headers_kwargs` with the values defined as follows:
        1. If the value in `headers_kwargs` is an array-like, it is indexed with the index of the currently processed
           header,
        2. Otherwise, it is kept unchanged.

        Examples
        --------
        >>> headers_kwargs = {
        ...     "headers": ["FirstBreakTrue", "FirstBreakPred"],
        ...     "s": 5,
        ...     "c": ["blue", "red"]
        ... }
        >>> Gather._parse_headers_kwargs(headers_kwargs, headers_key="headers")
        [{'headers': 'FirstBreakTrue', 's': 5, 'c': 'blue'},
         {'headers': 'FirstBreakPred', 's': 5, 'c': 'red'}]
        """
        if not isinstance(headers_kwargs, dict):
            return [{headers_key: header} for header in to_list(headers_kwargs)]

        if headers_key not in headers_kwargs:
            raise KeyError(f'{headers_key} key is not defined in event_headers')

        n_headers = len(to_list(headers_kwargs[headers_key]))
        kwargs_list = [{} for _ in range(n_headers)]
        for key, values in headers_kwargs.items():
            values = to_list(values)
            if len(values) == 1:
                values = values * n_headers
            elif len(values) != n_headers:
                raise ValueError(f"Incompatible length of {key} array: {n_headers} expected but {len(values)} given.")
            for ix, value in enumerate(values):
                kwargs_list[ix][key] = value
        return kwargs_list

    def _plot_headers(self, ax, headers_kwargs):
        """Add scatter plots of values of one or more headers over the main gather plot."""
        x_coords = np.arange(self.n_traces)
        kwargs_list = self._parse_headers_kwargs(headers_kwargs, "headers")
        for kwargs in kwargs_list:
            header = kwargs.pop("headers")
            label = kwargs.pop("label", header)
            process_outliers = kwargs.pop("process_outliers", "none")
            y_coords = self[header].ravel() / self.sample_rate
            if process_outliers == "clip":
                y_coords = np.clip(y_coords, 0, self.n_samples - 1)
            elif process_outliers == "discard":
                y_coords = np.where((y_coords >= 0) & (y_coords <= self.n_samples - 1), y_coords, np.nan)
            elif process_outliers != "none":
                raise ValueError(f"Unknown outlier processing mode {process_outliers}")
            ax.scatter(x_coords, y_coords, label=label, **kwargs)

        if headers_kwargs:
            ax.legend()

    def _plot_top_subplot(self, ax, divider, header_values, **kwargs):
        """Add a scatter plot of given header values on top of the main gather plot."""
        top_ax = divider.append_axes("top", sharex=ax, size="12%", pad=0.05)
        top_ax.scatter(np.arange(self.n_traces), header_values, **{"s": 5, "color": "black", **kwargs})
        top_ax.xaxis.set_visible(False)
        top_ax.yaxis.tick_right()
        top_ax.invert_yaxis()
        return top_ax

    def _get_x_ticks(self, axis_label):
        """Get tick labels for x-axis: either any gather header or ordinal numbers of traces in the gather."""
        if axis_label in self.headers.columns:
            return self[axis_label].reshape(-1)
        if axis_label == "index":
            return np.arange(self.n_traces)
        raise ValueError(f"Unknown label for x axis {axis_label}")

    def _get_y_ticks(self, axis_label):
        """Get tick labels for y-axis: either time samples or ordinal numbers of samples in the gather."""
        if axis_label == "time":
            return self.samples
        if axis_label == "samples":
            return np.arange(self.n_samples)
        raise ValueError(f"y axis label must be either `time` or `samples`, not {axis_label}")

    def _set_ticks(self, ax, axis, tick_src, ticker):
        """Set ticks, their labels and an axis label for a given axis."""
        # Get tick_labels depending on axis and its label
        if axis == "x":
            tick_labels = self._get_x_ticks(tick_src)
        elif axis == "y":
            tick_labels = self._get_y_ticks(tick_src)
        else:
            raise ValueError(f"Unknown axis {axis}")
        set_ticks(ax, axis, tick_labels=tick_labels, **{"label": tick_src, **ticker})
