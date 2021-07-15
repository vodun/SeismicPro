"""Implements Survey class describing a single SEG-Y file"""

import os
import warnings
from copy import copy, deepcopy
from textwrap import dedent

import segyio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.interpolate import interp1d

from .gather import Gather
from .utils import to_list, maybe_copy, calculate_stats, create_supergather_index


class Survey:
    """A class representing a single SEG-Y file.

    In order to reduce memory footprint, `Survey` instance does not store trace data, but only a requested subset of
    trace headers and general file meta such as `samples` and `sample_rate`. Trace data can be obtained by generating
    an instance of `Gather` class by calling either :func:`~Survey.get_gather` or :func:`~Survey.sample_gather`
    method.

    The resulting gather type depends on `header_index` argument, passed during `Survey` creation: traces are grouped
    into gathers by the common value of headers, defined by `header_index`. Some usual values of `header_index`
    include:
    - 'TRACE_SEQUENCE_FILE' - to get individual traces,
    - 'FieldRecord' - to get common source gathers,
    - ['GroupX', 'GroupY'] - to get common receiver gathers,
    - ['INLINE_3D', 'CROSSLINE_3D'] - to get common midpoint gathers.

    `header_cols` argument specifies all other trace headers to load to further be available in gather processing
    pipelines. Note that `TRACE_SEQUENCE_FILE` header is not loaded from the file but always automatically
    reconstructed. All loaded headers are stored in a `headers` attribute as a `pd.DataFrame` with `header_index`
    columns set as its index.

    Examples
    --------
    Create a survey of common source gathers and get a randomly selected gather from it:
    >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"])
    >>> gather = survey.sample_gather()

    Parameters
    ----------
    path : str
        A path to the source SEG-Y file.
    header_index : str or list of str
        Trace headers to be used to group traces into gathers.
    header_cols : str or list of str, optional
        Extra trace headers to load. If not given, only headers from `header_index` are loaded.
    name : str, optional
        Survey name. If not given, source file name is used. This name is mainly used to identify the survey when it is
        added to an index, see :class:`~index.SeismicIndex` docs for more info.
    limits : int or tuple or slice, optional
        Default time limits to be used during trace loading and survey statistics calculation. `int` or `tuple` are
        used as arguments to init a `slice` object. If not given, whole traces are used. Measured in samples.
    collect_stats : bool, optional, defaults to False
        Whether to calculate trace statistics for the survey.
    kwargs : misc, optional
        Additional keyword arguments to :func:`~Survey.collect_stats`.

    Attributes
    ----------
    path : str
        A path to the source SEG-Y file.
    name : str
        Survey name.
    headers : pd.DataFrame
        Loaded trace headers.
    samples : 1d np.ndarray of floats
        Recording time for each trace value. Measured in milliseconds.
    sample_rate : float
        Sample rate of seismic traces. Measured in milliseconds.
    limits : slice
        Default time limits to be used during trace loading and survey statistics calculation. Measured in samples.
    segy_handler : segyio.segy.SegyFile
        Source SEG-Y file handler.
    has_stats : bool
        Whether the survey has trace statistics calculated.
    min : np.float32
        Minimum trace value. Available only if trace statistics were calculated.
    max : np.float32
        Maximum trace value. Available only if trace statistics were calculated.
    mean : np.float32
        Mean trace value. Available only if trace statistics were calculated.
    std : np.float32
        Standard deviation of trace values. Available only if trace statistics were calculated.
    quantile_interpolator : scipy.interpolate.interp1d
        Trace values quantile interpolator. Available only if trace statistics were calculated.
    n_dead_traces : int
        The number of traces with constant value (dead traces). Available only if trace statistics were calculated.
    """
    def __init__(self, path, header_index, header_cols=None, name=None, limits=None, collect_stats=False, **kwargs):
        self.path = path
        basename = os.path.splitext(os.path.basename(self.path))[0]
        self.name = name if name is not None else basename

        if header_cols is None:
            header_cols = set()
        elif header_cols == "all":
            header_cols = set(segyio.tracefield.keys.keys())
        else:
            header_cols = set(to_list(header_cols))

        header_index = to_list(header_index)
        load_headers = set(header_index) | header_cols

        # We always reconstruct this column, so there is no need to load it.
        if "TRACE_SEQUENCE_FILE" in load_headers:
            load_headers.remove("TRACE_SEQUENCE_FILE")
            warn_msg = ("An automatically reconstructed TRACE_SEQUENCE_FILE header will be used instead of the one, "
                        f"contained in {basename}")
            warnings.warn(warn_msg, RuntimeWarning)

        self.segy_handler = segyio.open(self.path, ignore_geometry=True)
        self.segy_handler.mmap()

        # Get attributes from the source SEG-Y file.
        self.sample_rate = np.float32(segyio.dt(self.segy_handler) / 1000)
        self.file_samples = self.segy_handler.samples.astype(np.float32)

        # Set samples and samples_length according to passed `limits`.
        self.limits = None
        self.samples = None
        self.samples_length = None
        self.set_limits(limits)

        headers = {}
        for column in load_headers:
            headers[column] = self.segy_handler.attributes(segyio.tracefield.keys[column])[:]

        headers = pd.DataFrame(headers)
        # TRACE_SEQUENCE_FILE is reconstructed manually since it can be omitted according to the SEG-Y standard
        # but we rely on it during gather loading.
        headers["TRACE_SEQUENCE_FILE"] = np.arange(1, self.segy_handler.tracecount+1)
        headers.set_index(header_index, inplace=True)
        # Sort headers by index to optimize further headers subsampling and merging.
        self.headers = headers.sort_index()

        # Precalculate survey statistics if needed
        self.has_stats = False
        self.min = None
        self.max = None
        self.mean = None
        self.std = None
        self.quantile_interpolator = None
        self.n_dead_traces = None
        if collect_stats:
            self.collect_stats(**kwargs)

    @property
    def times(self):
        """1d np.ndarray of floats: Recording time for each trace value. Measured in milliseconds."""
        return self.samples

    def __del__(self):
        """Close SEG-Y file handler on survey destruction."""
        self.segy_handler.close()

    def __getstate__(self):
        """Create a survey's pickling state from its `__dict__` by setting SEG-Y file handler to `None`."""
        state = copy(self.__dict__)
        state["segy_handler"] = None
        return state

    def __setstate__(self, state):
        """Recreate a survey from unpickled state and reopen its source SEG-Y file."""
        self.__dict__ = state
        self.segy_handler = segyio.open(self.path, ignore_geometry=True)
        self.segy_handler.mmap()

    def __str__(self):
        """Print survey metadata including information about source file and trace statistics if they were
        calculated."""
        offsets = self.headers.get('offset')
        offset_range = f'[{np.min(offsets)} m, {np.max(offsets)} m]' if offsets is not None else None
        msg = f"""
        Survey path:               {self.path}
        Survey name:               {self.name}
        Survey size:               {os.path.getsize(self.path) / (1024**3):4.3f} GB
        Number of traces:          {self.headers.shape[0]}
        Traces length:             {self.samples_length} samples
        Sample rate:               {self.sample_rate} ms
        Offsets range:             {offset_range}

        Index name(s):             {', '.join(self.headers.index.names)}
        Number of unique indices:  {len(np.unique(self.headers.index))}
        """

        if self.has_stats:
            msg += f"""
        Survey statistics:
        Number of dead traces:     {self.n_dead_traces}
        Trace limits:              {self.limits.start}:{self.limits.stop}:{self.limits.step} samples
        mean | std:                {self.mean:>10.2f} | {self.std:<10.2f}
         min | max:                {self.min:>10.2f} | {self.max:<10.2f}
         q01 | q99:                {self.get_quantile(0.01):>10.2f} | {self.get_quantile(0.99):<10.2f}
        """
        return dedent(msg)

    def info(self):
        """Print survey metadata including information about source file and trace statistics if they were
        calculated."""
        print(self)

    #------------------------------------------------------------------------#
    #                     Statistics computation methods                     #
    #------------------------------------------------------------------------#

    def collect_stats(self, indices=None, n_quantile_traces=100000, quantile_precision=2, stats_limits=None, bar=True):
        """Collect the following trace data statistics by iterating over the survey:
        1. Min and max amplitude,
        2. Mean amplitude and trace standard deviation,
        3. Approximation of trace data quantiles with given precision,
        4. The number of dead traces.

        Since fair quantile calculation requires simultaneous loading of all traces from the file we avoid such memory
        overhead by calculating approximate quantiles for a small subset of `n_quantile_traces` traces selected
        randomly. Moreover, only a set of quantiles defined by `quantile_precision` is calculated, the rest of them are
        linearly interpolated by the collected ones.

        After the method is executed `has_stats` flag is set to `True` and all the calculated values can be obtained
        via corresponding attributes.

        Parameters
        ----------
        indices : pd.MultiIndex, optional
            A subset of survey headers indices to collect stats for. If not given, statistics are calculated for the
            whole survey.
        n_quantile_traces : positive int, optional, defaults to 100000
            The number of traces to use for quantiles estimation.
        quantile_precision : positive int, optional, defaults to 2
            Calculate an approximate quantile for each q with `quantile_precision` decimal places.
        stats_limits : int or tuple or slice, optional
            Time limits to be used for statistics calculation. `int` or `tuple` are used as arguments to init a `slice`
            object. If not given, whole traces are used. Measured in samples.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        survey : Survey
            The survey with collected stats. Sets `has_stats` flag to `True` and updates statistics attributes inplace.
        """
        headers = self.headers
        if indices is not None:
            headers = headers.loc[indices]
        traces_pos = headers.reset_index()["TRACE_SEQUENCE_FILE"].values - 1
        np.random.shuffle(traces_pos)

        limits = self.limits if stats_limits is None else self._process_limits(stats_limits)

        if n_quantile_traces <= 0:
            raise ValueError("n_quantile_traces must be positive")
        # Clip n_quantile_traces if it's greater than the total number of traces
        n_quantile_traces = min(n_quantile_traces, len(traces_pos))

        global_min, global_max = np.inf, -np.inf
        global_sum, global_sq_sum = 0, 0
        traces_length = 0
        self.n_dead_traces = 0

        traces_buf = np.empty((n_quantile_traces, self.samples_length), dtype=np.float32)
        trace = np.empty(self.samples_length, dtype=np.float32)

        # Accumulate min, max, mean and std values of survey traces
        for i, pos in tqdm(enumerate(traces_pos), desc=f"Calculating statistics for survey {self.name}",
                           total=len(traces_pos), disable=not bar):
            self.load_trace(buf=trace, index=pos, limits=limits, trace_length=self.samples_length)
            trace_min, trace_max, trace_sum, trace_sq_sum = calculate_stats(trace)
            global_min = min(trace_min, global_min)
            global_max = max(trace_max, global_max)
            global_sum += trace_sum
            global_sq_sum += trace_sq_sum
            traces_length += len(trace)
            self.n_dead_traces += np.isclose(trace_min, trace_max)

            # Sample random traces to calculate approximate quantiles
            if i < n_quantile_traces:
                traces_buf[i] = trace

        self.min = np.float32(global_min)
        self.max = np.float32(global_max)
        self.mean = np.float32(global_sum / traces_length)
        self.std = np.float32(np.sqrt((global_sq_sum / traces_length) - (global_sum / traces_length)**2))

        # Calculate all q-quantiles from 0 to 1 with step 1 / 10**quantile_precision
        q = np.round(np.linspace(0, 1, num=10**quantile_precision), decimals=quantile_precision)
        quantiles = np.nanquantile(traces_buf.ravel(), q=q)
        # 0 and 1 quantiles are replaced with actual min and max values respectively
        quantiles[0], quantiles[-1] = global_min, global_max
        self.quantile_interpolator = interp1d(q, quantiles)

        self.has_stats = True
        return self

    def get_quantile(self, q):
        """Calculate an approximation of the `q`-th quantile of the survey data.

        Notes
        -----
        Before calling this method, survey statistics must be calculated using `Survey.collect_stats`.

        Parameters
        ----------
        q : float or array-like of floats
            Quantile or a sequence of quantiles to compute, which must be between 0 and 1 inclusive.

        Returns
        -------
        quantile : float or array-like of floats
            Approximate `q`-th quantile values. Has the same type and shape as `q`.

        Raises
        ------
        ValueError
            If survey statistics were not calculated.
        """
        if not self.has_stats:
            raise ValueError('Global statistics were not calculated, call `Survey.collect_stats` first.')
        quantiles = self.quantile_interpolator(q).astype(np.float32)
        # return the same type as q: either single float or array-like
        return quantiles.item() if quantiles.ndim == 0 else quantiles

    #------------------------------------------------------------------------#
    #                            Loading methods                             #
    #------------------------------------------------------------------------#

    def load_gather(self, headers, limits=None, copy_headers=True):
        """Load gather by given headers.

        Parameters
        ----------
        headers : pd.DataFrame
            A dataframe of the traces to be loaded.
        limits : int or tuple or slice or None, optional, by default None
            Time limits for trace loading. `int` or `tuple` are used to construct a `slice` object directly. If not
            given, whole traces will be loaded. Measured in samples.
        copy_headers : bool, optional, by default True
            If True, the headers will be copied before passing into Gather, overwise headers passing as this.

        Returns
        -------
        gather : Gather
            Gather instance.
        """
        if copy_headers:
            headers = headers.copy()
        trace_indices = headers.reset_index()["TRACE_SEQUENCE_FILE"].values - 1

        limits = self.limits if limits is None else self._process_limits(limits)
        samples = self.file_samples[limits]
        trace_length = len(samples)

        data = np.empty((len(trace_indices), trace_length), dtype=np.float32)
        for i, ix in enumerate(trace_indices):
            self.load_trace(buf=data[i], index=ix, limits=limits, trace_length=trace_length)

        samples = self.file_samples[limits]
        sample_rate = np.float32(self.sample_rate * limits.step)
        gather = Gather(headers=headers, data=data, samples=samples, sample_rate=sample_rate, survey=self)
        return gather

    def get_gather(self, index, limits=None, copy_headers=True):
        """Load gather by specified `index`.

        Parameters
        ----------
        index : int or 1d array-like
            One of indices from `self.headers`.
        limits : int or tuple or slice or None, optional, by default None
            Time limits for trace loading. `int` or `tuple` are used to construct a `slice` object directly. If not
            given, whole traces will be loaded. Measured in samples.
        copy_headers : bool, optional, by default True
            If True, the headers will be copied before passing into Gather, overwise headers passing as this.

        Returns
        -------
        gather : Gather
            Gather instance.
        """
        gather_headers = self.headers.loc[index]
        # loc may sometimes return Series. In such cases slicing is used to guarantee, that DataFrame is returned
        if isinstance(gather_headers, pd.Series):
            gather_headers = self.headers.loc[index:index]
        return self.load_gather(gather_headers, limits, copy_headers)

    def sample_gather(self, limits=None, copy_headers=True):
        """Load gather with random index.

        Parameters
        ----------
        limits : int or tuple or slice or None, optional, by default None
            Time limits for trace loading. `int` or `tuple` are used to construct a `slice` object directly. If not
            given, whole traces will be loaded. Measured in samples.
        copy_headers : bool, optional, by default True
            If True, the headers will be copied before passing into Gather, overwise headers passing as this.

        Returns
        -------
        gather : Gather
            Gather instance.
        """
        index = np.random.choice(self.headers.index)
        gather = self.get_gather(index=index, limits=limits, copy_headers=copy_headers)
        return gather

    def load_trace(self, buf, index, limits, trace_length):
        """Load single trace with `index` position in SEG-Y file.

        To optimize trace loading process we use segyio's function `xfd.gettr`. We could not find a description of the
        input parameters, so we empirically determined their purpose:
            1. A buffer to write the loaded trace to,
            2. An index of the trace in a SEG-Y file to load,
            3. Unknown arg (always 1),
            4. Unknown arg (always 1),
            5. An index of the first trace element to load,
            6. An index of the last trace element to load,
            7. Trace element loading step,
            8. The overall number of samples to load.

        Parameters
        ----------
        buf : 1d np.ndarray of float32
            An empty array to save the loaded trace.
        index : int
            The position of the essential trace in the file.
        limits : slice
            Time slice for trace loading. Measured in samples.
        trace_length : int
            The length of loading trace

        Returns
        -------
        trace : 1d np.ndarray of float32
            Amplitudes of trace with `index` position.
        """
        return self.segy_handler.xfd.gettr(buf, index, 1, 1, limits.start, limits.stop, limits.step, trace_length)

    def load_first_breaks(self, path, first_breaks_col='FirstBreak'):
        """Load first breaks and save it to `self.headers`.

        File with first breaks data has three columns: FieldRecord, TraceNumber, FirstBreaks. The combination of
        FieldRecord and TraceNumber is a unique trace identifier. Thus we use it to match the value of the first breaks
        with the corresponding trace in `self.headers` and save the first break time to the column with name
        `first_breaks_col`.


        Parameters
        ----------
        path : str
            A path to the file with first breaks
        first_breaks_col : str, optional, by default 'FirstBreak'
            Column name in `self.headers` where the first breaks will be stored.

        Returns
        -------
        self : Survey
            Survey instance with first breaks column in headers.

        Raises
        ------
        ValueError
            If FieldRecord or TraceNumber are missed in `self.headers`.
            If there are no matches between columns ('FieldRecord', 'TraceNumber') in file with first breaks and in
            survey's headers.
        """
        segy_columns = ['FieldRecord', 'TraceNumber']
        first_breaks_columns = segy_columns + [first_breaks_col]
        first_breaks_df = pd.read_csv(path, names=first_breaks_columns, delim_whitespace=True, decimal=',')

        headers = self.headers.reset_index()
        missing_cols = set(segy_columns) - set(headers)
        if missing_cols:
            raise ValueError(f'Missing {missing_cols} column(s) required for first break loading.')

        headers = headers.merge(first_breaks_df, on=segy_columns)
        if headers.empty:
            raise ValueError('Empty headers after first breaks loading.')
        headers.set_index(self.headers.index.names, inplace=True)
        self.headers = headers.sort_index()
        return self

    #------------------------------------------------------------------------#
    #                       Survey processing methods                        #
    #------------------------------------------------------------------------#

    def copy(self):
        """Create a deepcopy of Survey instance.

        Returns
        -------
        survey : Survey
            Copy of Survey instance.
        """
        return deepcopy(self)

    @staticmethod
    def _apply(func, df, axis, unpack_args, **kwargs):
        """Apply function to `pd.DataFrame` along the specified axis.

        Parameters
        ----------
        func : callable
            Function that will be applied to `df`.
        df : pd.DataFrame
            DataFrame to which the function is applied.
        axis : int or None
            A dimension by which the function is applied.
        unpack_args : bool
            If True, given aruments will be unpacked before passing to the `pd.DataFrame.apply` function, otherwise
            aruments will passed as this.
        kwargs : misc, optional
            Additional keyword arguments to pass as keywords arguments to `func` or `pd.DataFrame.apply`.

        Returns
        -------
        result : np.ndarray
            The result of applying `func` to `df` by specified axis.
        """
        if axis is None:
            res = func(df, **kwargs)
        else:
            apply_func = lambda args: func(*args) if unpack_args else func
            res = df.apply(apply_func, axis=axis, raw=True, **kwargs)
        return res.values

    def filter(self, cond, cols, axis=None, unpack_args=False, inplace=False, **kwargs):
        """Filter `self.headers` by certain condition along the specified axis.

        Parameters
        ----------
        cond : callable
            Function that will be applied to `self.headers` and must return boolean mask with `True` for elements that
            match the condition, and `False` for elements that don't.
        cols : str or list of str
            Columns to which condition is applied.
        axis : int or None, optional, by default None
            A dimension by which the condition is applied.
        unpack_args : bool, optional, by default False
            If True, given aruments will be unpacked before passing to the `pd.DataFrame.apply` function, otherwise
            aruments will passed as this.
        inplace : bool, optional, by default False
            If True, `self.headers` will be filtered inplace, otherwise copy of the Survey is filtered.
        kwargs : misc, optional
            Additional keyword arguments to pass as keywords arguments to `cond` or `pd.DataFrame.apply`.

        Returns
        -------
        self : Survey
            Fileterd Survey.

        Raises
        ------
        ValueError
            If `cond` is returns more than one bool value for each header row.
        """
        self = maybe_copy(self, inplace)
        headers = self.headers.reset_index()[to_list(cols)]
        mask = self._apply(cond, headers, axis=axis, unpack_args=unpack_args, **kwargs)
        if (mask.ndim == 2) and (mask.shape[1] == 1):
            mask = mask[:, 0]
        if mask.ndim != 1:
            raise ValueError("cond must return a single bool value for each header row")
        self.headers = self.headers.loc[mask]
        return self

    def apply(self, func, cols, res_cols=None, axis=None, unpack_args=False, inplace=False, **kwargs):
        """Apply function to `self.headers` along the specified axis.

        Parameters
        ----------
        func : callable
            Function that will be applied to `self.headers` and must return boolean mask with `True` for elements that
            match the condition, and `False` for elements that don't.
        cols : str or list of str
            Columns to which condition is applied.
        res_cols : str or list of str, optional, by default None
            Columns to which result is saved.
        axis : int or None, optional, by default None
            A dimension by which the condition is applied.
        unpack_args : bool, optional, by default False
            If True, given aruments will be unpacked before passing to the `pd.DataFrame.apply` function, otherwise
            aruments will passed as this.
        inplace : bool, optional, by default False
            If True, the result will be saved into `self.headers` of certain Survey, otherwise copy of the Survey will
            be created before apply operation.
        kwargs : misc, optional
            Additional keyword arguments to pass as keywords arguments to `func` or `pd.DataFrame.apply`.

        Returns
        -------
        self : Survey
            Survey with new column(s) `res_cols`.
        """
        self = maybe_copy(self, inplace)
        cols = to_list(cols)
        res_cols = cols if res_cols is None else to_list(res_cols)
        headers = self.headers.reset_index()[cols]
        res = self._apply(func, headers, axis=axis, unpack_args=unpack_args, **kwargs)
        self.headers[res_cols] = res
        return self

    def reindex(self, new_index, inplace=False):
        """Change the index in `self.headers` to `new_index`.

        Parameters
        ----------
        new_index : str or list of str
            Column(s) name that will be the new index.
        inplace : bool, optional, by default False
            If True, `self.headers` will be filtered inplace, otherwise copy of the Survey is filtered.

        Returns
        -------
        self : Survey
            Survey with new index.
        """
        self = maybe_copy(self, inplace)
        self.headers.reset_index(inplace=True)
        self.headers.set_index(new_index, inplace=True)
        self.headers.sort_index(inplace=True)
        return self

    def set_limits(self, limits):
        """Convert limits to slice and save it to `self.limits`.

        These limits are default time limits to be used during trace loading and survey statistics calculation.

        Parameters
        ----------
        limits : int or tuple or slice, optional
            Time limits. `int` or `tuple` are used to construct a `slice` object directly. Measured in samples.
            The resulted slice contains three numbers:
            * start sample position during trace loading,
            * end sample position during trace loading,
            * step.

        Raises
        ------
        ValueError
            If resulted trace's length is zero.
        """
        self.limits = self._process_limits(limits)
        self.samples = self.file_samples[self.limits]
        self.samples_length = len(self.samples)
        if self.samples_length == 0:
            raise ValueError('Trace length must be positive.')

    def _process_limits(self, limits):
        """Convert limits to slice."""
        if not isinstance(limits, slice):
            limits = slice(*to_list(limits))
        # Use .indices to avoid negative slicing range
        limits = limits.indices(len(self.file_samples))
        if limits[-1] < 0:
            raise ValueError('Negative step is not allowed.')
        return slice(*limits)

    #------------------------------------------------------------------------#
    #                         Task specific methods                          #
    #------------------------------------------------------------------------#

    def generate_supergathers(self, size=(3, 3), step=(20, 20), modulo=(0, 0), reindex=True, inplace=False):
        """Combine several adjacent CDP gathers into ensembles called supergathers.

        Supergather generation is usually performed as a first step of velocity analysis. A substantially larger number
        of traces processed at once leads to increased signal-to-noise ratio: seismic wave reflections are much more
        clearly visible than on single CDP gathers and the velocity spectra calculated using
        :func:`~Gather.calculate_semblance` are more coherent which allow for more accurate stacking velocity picking.

        The method creates two new `headers` columns called `SUPERGATHER_INLINE_3D` and `SUPERGATHER_CROSSLINE_3D`
        equal to `INLINE_3D` and `CROSSLINE_3D` of the central CDP gather. Note, that some gathers may be assigned to
        several supergathers at once and their traces will become duplicated.

        Parameters
        ----------
        size : tuple of 2 ints, optional, defaults to (3, 3)
            Supergather size along inline and crossline axes. Measured in lines.
        step : tuple of 2 ints, optional, defaults to (20, 20)
            Supergather step along inline and crossline axes. Measured in lines.
        modulo : tuple of 2 ints, optional, defaults to (0, 0)
            The remainder of the division of gather coordinates by given `step` for it to become a supergather center.
            Used to shift the grid of supergathers from the field origin. Measured in lines.
        reindex : bool, optional, defaults to True
            Whether to reindex a survey with the created `SUPERGATHER_INLINE_3D` and `SUPERGATHER_CROSSLINE_3D` headers
            columns.
        inplace : bool, optional, defaults to False
            Whether to transform the survey inplace or return a new one.

        Returns
        -------
        survey : Survey
            A survey with generated supergathers.

        Raises
        ------
        KeyError
            If `INLINE_3D` and `CROSSLINE_3D` headers were not loaded.
        """
        self = maybe_copy(self, inplace)
        index_cols = self.headers.index.names
        headers = self.headers.reset_index()
        line_cols = ["INLINE_3D", "CROSSLINE_3D"]
        super_line_cols = ["SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"]

        if any(col not in headers for col in line_cols):
            raise KeyError("INLINE_3D and CROSSLINE_3D headers are not loaded")
        supergather_centers_mask = ((headers["INLINE_3D"] % step[0] == modulo[0]) &
                                    (headers["CROSSLINE_3D"] % step[1] == modulo[1]))
        supergather_centers = headers.loc[supergather_centers_mask, line_cols]
        supergather_centers = supergather_centers.drop_duplicates().sort_values(by=line_cols)
        supergather_lines = pd.DataFrame(create_supergather_index(supergather_centers.values, size),
                                         columns=super_line_cols+line_cols)
        self.headers = pd.merge(supergather_lines, headers, on=line_cols)

        if reindex:
            index_cols = super_line_cols
        self.headers.set_index(index_cols, inplace=True)
        self.headers.sort_index(inplace=True)
        return self
