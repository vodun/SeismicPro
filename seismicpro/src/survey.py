"""Implements Survey class describing a single SEG-Y file"""

# pylint: disable=self-cls-assignment
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
from .custom_headers import HDR_DEAD_TRACE


class Survey:  # pylint: disable=too-many-instance-attributes
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
        Extra trace headers to load. If not given, only headers from `header_index` are loaded and a
        `TRACE_SEQUENCE_FILE` header is created automatically.
    name : str, optional
        Survey name. If not given, source file name is used. This name is mainly used to identify the survey when it is
        added to an index, see :class:`~index.SeismicIndex` docs for more info.
    limits : int or tuple or slice, optional
        Default time limits to be used during trace loading and survey statistics calculation. `int` or `tuple` are
        used as arguments to init a `slice` object. If not given, whole traces are used. Measured in samples.

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
        The number of traces with constant value (dead traces). None until `mark_dead_traces` is called.
    """
    def __init__(self, path, header_index, header_cols=None, name=None, limits=None):
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
        self.file_samples = self.segy_handler.samples.astype(np.float32)
        self.file_sample_rate = np.float32(segyio.dt(self.segy_handler) / 1000)

        # Set samples and sample_rate according to passed `limits`.
        self.limits = None
        self.samples = None
        self.sample_rate = None
        self.set_limits(limits)

        headers = {}
        for column in load_headers:
            headers[column] = self.segy_handler.attributes(segyio.tracefield.keys[column])[:]

        # According to SEG-Y spec, headers values are at most 4-byte integers
        headers = pd.DataFrame(headers, dtype=np.int32)
        # TRACE_SEQUENCE_FILE is reconstructed manually since it can be omitted according to the SEG-Y standard
        # but we rely on it during gather loading.
        tsf_dtype = np.int32 if len(headers) < np.iinfo(np.int32).max else np.int64
        headers["TRACE_SEQUENCE_FILE"] = np.arange(1, self.segy_handler.tracecount+1, dtype=tsf_dtype)
        headers.set_index(header_index, inplace=True)

        # Sort headers by index to optimize further headers subsampling and merging.
        # Preserve trace order from the file for traces from the same gather.
        self.headers = headers.sort_index(kind="stable")

        # Precalculate survey statistics if needed
        self.has_stats = False
        self.min = None
        self.max = None
        self.mean = None
        self.std = None
        self.quantile_interpolator = None
        self.n_dead_traces = None

    @property
    def times(self):
        """1d np.ndarray of floats: Recording time for each trace value. Measured in milliseconds."""
        return self.samples

    @property
    def n_traces(self):
        """int: The number of traces in the survey."""
        return len(self.headers)

    @property
    def n_samples(self):
        """int: Trace length in samples."""
        return len(self.samples)

    @property
    def n_file_samples(self):
        """int: Trace length in samples in the source SEG-Y file."""
        return len(self.file_samples)

    @property
    def traces_checked(self):
        """bool: `mark_dead_traces` called"""
        return self.n_dead_traces is not None

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

        Number of traces:          {self.n_traces}
        Trace length:              {self.n_samples} samples
        Sample rate:               {self.sample_rate} ms
        Times range:               [{min(self.samples)} ms, {max(self.samples)} ms]
        Offsets range:             {offset_range}

        Index name(s):             {', '.join(self.headers.index.names)}
        Number of unique indices:  {len(np.unique(self.headers.index))}
        """

        if self.has_stats:
            msg += f"""
        Survey statistics:
        mean | std:                {self.mean:>10.2f} | {self.std:<10.2f}
         min | max:                {self.min:>10.2f} | {self.max:<10.2f}
         q01 | q99:                {self.get_quantile(0.01):>10.2f} | {self.get_quantile(0.99):<10.2f}
        """

        if self.traces_checked:
            msg += f"""
        Number of dead traces:     {self.n_dead_traces}
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
        """Collect the following statistics by iterating over non-dead traces of the survey:
        1. Min and max amplitude,
        2. Mean amplitude and trace standard deviation,
        3. Approximation of trace data quantiles with given precision.

        Since fair quantile calculation requires simultaneous loading of all traces from the file we avoid such memory
        overhead by calculating approximate quantiles for a small subset of `n_quantile_traces` traces selected
        randomly. Only a set of quantiles defined by `quantile_precision` is calculated, the rest of them are linearly
        interpolated by the collected ones.

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
            Calculate an approximate quantile for each q with `quantile_precision` decimal places. All other quantiles
            will be linearly interpolated on request.
        stats_limits : int or tuple or slice, optional
            Time limits to be used for statistics calculation. `int` or `tuple` are used as arguments to init a `slice`
            object. If not given, whole traces are used. Measured in samples.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        survey : Survey
            The survey with collected stats. Sets `has_stats` flag to `True`, and updates statistics attributes inplace.
        """
        if not self.traces_checked:
            warn_msg = ("Dead traces detection was not performed, collected statistics may be skewed. "
                        "Run `mark_dead_traces`to remove this warning")
            warnings.warn(warn_msg, RuntimeWarning)

        headers = self.headers
        if indices is not None:
            headers = headers.loc[indices]
        n_traces = len(headers)
        traces_pos = headers.reset_index()["TRACE_SEQUENCE_FILE"].values - 1
        np.random.shuffle(traces_pos)

        limits = self.limits if stats_limits is None else self._process_limits(stats_limits)
        n_samples = len(self.file_samples[limits])

        if n_quantile_traces < 0:
            raise ValueError("n_quantile_traces must be non-negative")
        # Clip n_quantile_traces if it's greater than the total number of traces
        n_quantile_traces = min(n_quantile_traces, n_traces)

        global_min, global_max = np.inf, -np.inf
        global_sum, global_sq_sum = 0, 0
        traces_length = 0

        traces_buf = np.empty((n_quantile_traces, n_samples), dtype=np.float32)
        trace = np.empty(n_samples, dtype=np.float32)

        quantile_traces_counter = 0
        # Accumulate min, max, mean and std values of survey traces
        for pos in tqdm(traces_pos, desc=f"Calculating statistics for survey {self.name}",
                      total=n_traces, disable=not bar):
            self.load_trace(buf=trace, index=pos, limits=limits, trace_length=n_samples)
            trace_min, trace_max, trace_sum, trace_sq_sum = calculate_stats(trace)

            global_min = min(trace_min, global_min)
            global_max = max(trace_max, global_max)
            global_sum += trace_sum
            global_sq_sum += trace_sq_sum
            traces_length += len(trace)

            # Sample random traces to calculate approximate quantiles
            if quantile_traces_counter < n_quantile_traces:
                traces_buf[quantile_traces_counter] = trace
                quantile_traces_counter += 1


        self.min = np.float32(global_min)
        self.max = np.float32(global_max)
        self.mean = np.float32(global_sum / traces_length)
        self.std = np.float32(np.sqrt((global_sq_sum / traces_length) - (global_sum / traces_length)**2))

        if n_quantile_traces == 0:
            q = [0, 1]
            quantiles = [global_min, global_max]
        else:
            traces_buf = traces_buf[:n_quantile_traces]
            # Calculate all q-quantiles from 0 to 1 with step 1 / 10**quantile_precision
            q = np.round(np.linspace(0, 1, num=10**quantile_precision), decimals=quantile_precision)
            quantiles = np.nanquantile(traces_buf.ravel(), q=q)
            # 0 and 1 quantiles are replaced with actual min and max values respectively
            quantiles[0], quantiles[-1] = global_min, global_max
        self.quantile_interpolator = interp1d(q, quantiles)

        self.has_stats = True
        return self

    def mark_dead_traces(self, bar=True):
        """Mark dead traces (those having constant amplitudes) by setting a value of a new `DeadTrace`
        header to `True` and stores the overall number of dead traces in the `n_dead_traces` attribute.

        Parameters
        ----------
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        survey : Survey
            The same survey with a new `DeadTrace` header created.
        """

        traces_pos = self.headers.reset_index()["TRACE_SEQUENCE_FILE"].values - 1
        n_samples = len(self.file_samples[self.limits])

        trace = np.empty(n_samples, dtype=np.float32)
        dead_indices = []
        for pos in tqdm(traces_pos, desc=f"Detecting dead traces for survey {self.name}",
                      total=len(self.headers), disable=not bar):
            self.load_trace(buf=trace, index=pos, limits=self.limits, trace_length=n_samples)
            trace_min, trace_max, *_ = calculate_stats(trace)

            if np.isclose(trace_min, trace_max):
                dead_indices.append(pos)

        self.n_dead_traces = len(dead_indices)
        self.headers[HDR_DEAD_TRACE] = False
        self.headers.iloc[dead_indices, self.headers.columns.get_loc(HDR_DEAD_TRACE)] = True


    def get_quantile(self, q):
        """Calculate an approximation of the `q`-th quantile of the survey data.

        Notes
        -----
        Before calling this method, survey statistics must be calculated using :func:`~Survey.collect_stats`.

        Parameters
        ----------
        q : float or array-like of floats
            Quantile or a sequence of quantiles to compute, which must be between 0 and 1 inclusive.

        Returns
        -------
        quantile : float or array-like of floats
            Approximate `q`-th quantile values. Has the same shape as `q`.

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
        """Load a gather with given `headers`.

        Parameters
        ----------
        headers : pd.DataFrame
            Headers of traces to load. Must be a subset of `self.headers`.
        limits : int or tuple or slice or None, optional, defaults to None
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, whole traces are loaded. Measured in samples.
        copy_headers : bool, optional, defaults to True
            Whether to copy the passed `headers` when instantiating the gather.

        Returns
        -------
        gather : Gather
            Loaded gather instance.
        """
        if copy_headers:
            headers = headers.copy()
        trace_indices = headers.reset_index()["TRACE_SEQUENCE_FILE"].values - 1

        limits = self.limits if limits is None else self._process_limits(limits)
        samples = self.file_samples[limits]
        sample_rate = np.float32(self.file_sample_rate * limits.step)
        n_samples = len(samples)

        data = np.empty((len(trace_indices), n_samples), dtype=np.float32)
        for i, ix in enumerate(trace_indices):
            self.load_trace(buf=data[i], index=ix, limits=limits, trace_length=n_samples)

        gather = Gather(headers=headers, data=data, samples=samples, sample_rate=sample_rate, survey=self)
        return gather

    def get_gather(self, index, limits=None, copy_headers=True):
        """Load a gather with given `index`.

        Parameters
        ----------
        index : int or 1d array-like
            An index of the gather to load. Must be one of `self.headers.index`.
        limits : int or tuple or slice or None, optional, defaults to None
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, whole traces are loaded. Measured in samples.
        copy_headers : bool, optional, defaults to True
            Whether to copy the subset of survey `headers` describing the gather.

        Returns
        -------
        gather : Gather
            Loaded gather instance.
        """
        gather_headers = self.headers.loc[index]
        # loc may sometimes return Series. In such cases slicing is used to guarantee, that DataFrame is returned
        if isinstance(gather_headers, pd.Series):
            gather_headers = self.headers.loc[index:index]
        return self.load_gather(gather_headers, limits, copy_headers)

    def sample_gather(self, limits=None, copy_headers=True):
        """Load a gather with random index.

        Parameters
        ----------
        limits : int or tuple or slice or None, optional, defaults to None
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, whole traces are loaded. Measured in samples.
        copy_headers : bool, optional, defaults to True
            Whether to copy the subset of survey `headers` describing the sampled gather.

        Returns
        -------
        gather : Gather
            Loaded gather instance.
        """
        index = np.random.choice(self.headers.index)
        gather = self.get_gather(index=index, limits=limits, copy_headers=copy_headers)
        return gather

    def load_trace(self, buf, index, limits, trace_length):
        """Load a single trace from a SEG-Y file by its position.

        In order to optimize trace loading process, we use `segyio`'s low-level function `xfd.gettr`. Description of
        its arguments is given below:
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
            Trace position in the file.
        limits : slice
            Trace time range to load. Measured in samples.
        trace_length : int
            Total number of samples to load.

        Returns
        -------
        trace : 1d np.ndarray of float32
            Loaded trace.
        """
        return self.segy_handler.xfd.gettr(buf, index, 1, 1, limits.start, limits.stop, limits.step, trace_length)

    # pylint: disable=anomalous-backslash-in-string
    def load_first_breaks(self, path, trace_id_cols=('FieldRecord', 'TraceNumber'), first_breaks_col='FirstBreak',
                          delimiter='\s+', decimal=None, encoding="UTF-8", inplace=False, **kwargs):
        """Load times of first breaks from a file and save them to a new column in headers.

        Each line of the file stores the first break time for a trace in the last column.
        The combination of all but the last columns should act as a unique trace identifier and is used to match
        the trace from the file with the corresponding trace in `self.headers`.

        The file can have any format that can be read by `pd.read_csv`, by default, it's expected
        to have whitespace-separated values.

        Parameters
        ----------
        path : str
            A path to the file with first break times in milliseconds.
        trace_id_cols : tuple of str, defaults to ('FieldRecord', 'TraceNumber')
            Headers, whose values are stored in all but the last columns of the file.
        first_breaks_col : str, optional, defaults to 'FirstBreak'
            Column name in `self.headers` where loaded first break times will be stored.
        delimiter: str, defaults to '\s+'
            Delimiter to use. See `pd.read_csv` for more details.
        decimal : str, defaults to None
            Character to recognize as decimal point.
            If `None`, it is inferred from the first line of the file.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.
        inplace : bool, optional, defaults to False
            Whether to load first break times inplace or to a survey copy.
        kwargs : misc, optional
            Additional keyword arguments to pass to `pd.read_csv`.

        Returns
        -------
        self : Survey
            A survey with loaded first break times. Changes `self.headers` inplace.

        Raises
        ------
        ValueError
            If there is not a single match of rows from the file with those in `self.headers`.
        """
        self = maybe_copy(self, inplace)

        # if decimal is not provided, try to infer it from the first line
        if decimal is None:
            with open(path, 'r', encoding=encoding) as f:
                row = f.readline()
            decimal = '.' if '.' in row else ','

        file_columns = to_list(trace_id_cols) + [first_breaks_col]
        first_breaks_df = pd.read_csv(path, delimiter=delimiter, names=file_columns,
                                      decimal=decimal, encoding=encoding, **kwargs)

        headers = self.headers.reset_index()
        headers = headers.merge(first_breaks_df, on=trace_id_cols)
        if headers.empty:
            raise ValueError('Empty headers after first breaks loading.')
        headers.set_index(self.headers.index.names, inplace=True)
        self.headers = headers.sort_index(kind="stable")
        return self

    #------------------------------------------------------------------------#
    #                       Survey processing methods                        #
    #------------------------------------------------------------------------#

    def copy(self):
        """Create a deepcopy of a `Survey` instance.

        Returns
        -------
        survey : Survey
            Survey copy.
        """
        return deepcopy(self)

    @staticmethod
    def _apply(func, df, axis, unpack_args, **kwargs):
        """Apply a function to a `pd.DataFrame` along the specified axis.

        Parameters
        ----------
        func : callable
            A function to be applied to `df`.
        df : pd.DataFrame
            A `DataFrame` to which the function will be applied.
        axis : {0 or "index", 1 or "columns", None}
            An axis along which the function is applied:
            - 0 or "index": apply a function to each column,
            - 1 or "columns": apply a function to each row,
            - `None`: apply a function to the `DataFrame` as a whole.
        unpack_args : bool
            If `True`, row or column values are passed to `func` as individual arguments, otherwise the whole array is
            passed as a single arg. If `axis` is `None` and `unpack_args` is `True`, columns of the `df` are passed to
            the `func` as individual arguments.
        kwargs : misc, optional
            Additional keyword arguments to pass to `func` or `pd.DataFrame.apply`.

        Returns
        -------
        result : np.ndarray
            The result of applying `func` to `df`.
        """
        if axis is None:
            args = (col_val for _, col_val in df.iteritems()) if unpack_args else (df,)
            res = func(*args, **kwargs)
        else:
            # FIXME: Workaround for a pandas bug https://github.com/pandas-dev/pandas/issues/34822
            # raw=True causes incorrect apply behavior when axis=1 and several values are returned from `func`
            raw = (axis != 1)

            apply_func = (lambda args, **kwargs: func(*args, **kwargs)) if unpack_args else func
            res = df.apply(apply_func, axis=axis, raw=raw, result_type="expand", **kwargs)

        if isinstance(res, pd.Series):
            res = res.to_frame()
        return res.values

    def filter(self, cond, cols, axis=None, unpack_args=False, inplace=False, **kwargs):
        """Keep only those rows of `headers` where `cond` is `True`.

        Examples
        --------
        Keep only traces whose offset is less than 1500 meters:
        >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
        >>> survey.filter(lambda offset: offset < 1500, cols="offset", inplace=True)

        Parameters
        ----------
        cond : callable
            A function to be applied to `self.headers` to get a filtering mask. Must return a boolean array whose
            length equals to the length of `headers` and `True` values correspond to traces to keep.
        cols : str or list of str
            `self.headers` columns for which condition is checked.
        axis : {0 or "index", 1 or "columns", None}, optional, defaults to None
            An axis along which `cond` is applied:
            - 0 or "index": apply `cond` to each column,
            - 1 or "columns": apply `cond` to each row,
            - `None`: apply `cond` to the `DataFrame` as a whole.
        unpack_args : bool, optional, defaults to False
            If `True`, row or column values are passed to `cond` as individual arguments, otherwise the whole array is
            passed as a single arg. If `axis` is `None` and `unpack_args` is `True`, each column from `cols` is passed
            to the `cond` as an individual argument.
        inplace : bool, optional, defaults to False
            Whether to perform filtering inplace or process a survey copy.
        kwargs : misc, optional
            Additional keyword arguments to pass to `cond` or `pd.DataFrame.apply`.

        Returns
        -------
        self : Survey
            Filtered survey.

        Raises
        ------
        ValueError
            If `cond` returns more than one bool value for each row of `headers`.
        """
        self = maybe_copy(self, inplace)
        headers = self.headers.reset_index()[to_list(cols)]
        mask = self._apply(cond, headers, axis=axis, unpack_args=unpack_args, **kwargs)
        if (mask.ndim != 2) or (mask.shape[1] != 1):
            raise ValueError("cond must return a single value for each header row")
        if mask.dtype != np.bool_:
            raise ValueError("cond must return a bool value for each header row")
        self.headers = self.headers.loc[mask[:, 0]]
        return self

    def apply(self, func, cols, res_cols=None, axis=None, unpack_args=False, inplace=False, **kwargs):
        """Apply a function to `self.headers` along the specified axis.

        Examples
        --------
        Convert signed offsets to their absolute values:
        >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
        >>> survey.apply(lambda offset: np.abs(offset), cols="offset", inplace=True)

        Parameters
        ----------
        func : callable
            A function to be applied to `self.headers`. Must return a 2d object with shape (`len(self.headers)`,
            `len(res_cols)`).
        cols : str or list of str
            `self.headers` columns for which the function is applied.
        res_cols : str or list of str, optional, defaults to None
            `self.headers` columns in which the result is saved. If not given, equals to `cols`.
        axis : {0 or "index", 1 or "columns", None}, optional, defaults to None
            An axis along which the function is applied:
            - 0 or "index": apply a function to each column,
            - 1 or "columns": apply a function to each row,
            - `None`: apply a function to the `DataFrame` as a whole.
        unpack_args : bool, optional, defaults to False
            If `True`, row or column values are passed to `func` as individual arguments, otherwise the whole array is
            passed as a single arg. If `axis` is `None` and `unpack_args` is `True`, each column from `cols` is passed
            to the `func` as an individual argument.
        inplace : bool, optional, defaults to False
            Whether to apply the function inplace or to a survey copy.
        kwargs : misc, optional
            Additional keyword arguments to pass to `func` or `pd.DataFrame.apply`.

        Returns
        -------
        self : Survey
            A survey with the function applied.
        """
        self = maybe_copy(self, inplace)
        cols = to_list(cols)
        res_cols = cols if res_cols is None else to_list(res_cols)
        headers = self.headers.reset_index()[cols]
        res = self._apply(func, headers, axis=axis, unpack_args=unpack_args, **kwargs)
        self.headers[res_cols] = res
        return self

    def reindex(self, new_index, inplace=False):
        """Change the index of `self.headers` to `new_index`.

        Parameters
        ----------
        new_index : str or list of str
            Headers columns to become a new index.
        inplace : bool, optional, defaults to False
            Whether to perform reindexation inplace or return a new survey instance.

        Returns
        -------
        self : Survey
            Reindexed survey.
        """
        self = maybe_copy(self, inplace)
        self.headers.reset_index(inplace=True)
        self.headers.set_index(new_index, inplace=True)
        self.headers.sort_index(kind="stable", inplace=True)
        return self

    def set_limits(self, limits):
        """Update default survey time limits that are used during trace loading and statistics calculation.

        Parameters
        ----------
        limits : int or tuple or slice
            Default time limits to be used during trace loading and survey statistics calculation. `int` or `tuple` are
            used as arguments to init a `slice`. The resulting object is stored in `self.limits` attribute and used to
            recalculate `self.samples` and `self.sample_rate`. Measured in samples.

        Raises
        ------
        ValueError
            If negative step of limits was passed.
            If the resulting samples length is zero.
        """
        self.limits = self._process_limits(limits)
        self.samples = self.file_samples[self.limits]
        self.sample_rate = self.file_sample_rate * self.limits.step

    def _process_limits(self, limits):
        """Convert given `limits` to a `slice`."""
        if not isinstance(limits, slice):
            limits = slice(*to_list(limits))
        # Use .indices to avoid negative slicing range
        limits = limits.indices(len(self.file_samples))
        if limits[-1] < 0:
            raise ValueError('Negative step is not allowed.')
        if limits[1] <= limits[0]:
            raise ValueError('Empty traces after setting limits.')
        return slice(*limits)

    def remove_dead_traces(self, inplace=False, bar=True):
        """ Removes dead (constant) traces from survey's data.
        Calls `mark_dead_traces` if it was not called before.

        Parameters
        ----------
        inplace : bool, optional, defaults to False
            Whether to remove traces inplace or return a new survey instance.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        Survey
            Survey with no dead traces.
        """
        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        if not self.traces_checked:
            self.mark_dead_traces(bar)

        self.filter(lambda dt: ~dt, cols=HDR_DEAD_TRACE, inplace=True)
        self.n_dead_traces = 0

        return self

    #------------------------------------------------------------------------#
    #                         Task specific methods                          #
    #------------------------------------------------------------------------#

    def generate_supergathers(self, size=(3, 3), step=(20, 20), modulo=(0, 0), reindex=True, inplace=False):
        """Combine several adjacent CDP gathers into ensembles called supergathers.

        Supergather generation is usually performed as a first step of velocity analysis. A substantially larger number
        of traces processed at once leads to increased signal-to-noise ratio: seismic wave reflections are much more
        clearly visible than on single CDP gathers and the velocity spectra calculated using
        :func:`~Gather.calculate_semblance` are more coherent which allows for more accurate stacking velocity picking.

        The method creates two new `headers` columns called `SUPERGATHER_INLINE_3D` and `SUPERGATHER_CROSSLINE_3D`
        equal to `INLINE_3D` and `CROSSLINE_3D` of the central CDP gather. Note, that some gathers may be assigned to
        several supergathers at once and their traces will become duplicated in `headers`.

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
            Whether to transform the survey inplace or process its copy.

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
        self.headers.sort_index(kind="stable", inplace=True)
        return self
