"""Implements Survey class describing a single SEG-Y file"""

import os
import warnings
from copy import copy
from textwrap import dedent
import math

import cv2
import segyio
import numpy as np
import scipy as sp
import pandas as pd
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from .headers import load_headers
from .metrics import SurveyAttribute
from .plot_geometry import SurveyGeometryPlot
from .utils import ibm_to_ieee, calculate_trace_stats
from ..gather import Gather
from ..metrics import PartialMetric
from ..containers import GatherContainer, SamplesContainer
from ..utils import to_list, maybe_copy, get_cols, int_linspace
from ..const import ENDIANNESS, HDR_DEAD_TRACE, HDR_FIRST_BREAK


class Survey(GatherContainer, SamplesContainer):  # pylint: disable=too-many-instance-attributes
    """A class representing a single SEG-Y file.

    In order to reduce memory footprint, `Survey` instance does not store trace data, but only a requested subset of
    trace headers and general file meta such as `samples` and `sample_rate`. Trace data can be obtained by generating
    an instance of `Gather` class by calling either :func:`~Survey.get_gather` or :func:`~Survey.sample_gather`
    method.

    The resulting gather type depends on `header_index` argument passed during `Survey` creation: traces are grouped
    into gathers by the common value of headers, defined by `header_index`. Some frequently used values of
    `header_index` are:
    - 'TRACE_SEQUENCE_FILE' - to get individual traces,
    - 'FieldRecord' - to get common source gathers,
    - ['GroupX', 'GroupY'] - to get common receiver gathers,
    - ['INLINE_3D', 'CROSSLINE_3D'] - to get common midpoint gathers.

    `header_cols` argument specifies all other trace headers to load to further be available in gather processing
    pipelines. All loaded headers are stored in a `headers` attribute as a `pd.DataFrame` with `header_index` columns
    set as its index.

    Values of both `header_index` and `header_cols` must be any of those specified in
    https://segyio.readthedocs.io/en/latest/segyio.html#trace-header-keys except for `UnassignedInt1` and
    `UnassignedInt2` since they are treated differently from all other headers by `segyio`. Also, `TRACE_SEQUENCE_FILE`
    header is not loaded from the file but always automatically reconstructed.

    The survey sample rate is calculated by two values stored in:
    * bytes 3217-3218 of the binary header, called `Interval` in `segyio`,
    * bytes 117-118 of the trace header of the first trace in the file, called `TRACE_SAMPLE_INTERVAL` in `segyio`.
    If both of them are present and equal or only one of them is well-defined (non-zero), it is used as a sample rate.
    Otherwise, an error is raised.

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
        Trace headers to be used to group traces into gathers. Must be any of those specified in
        https://segyio.readthedocs.io/en/latest/segyio.html#trace-header-keys except for `UnassignedInt1` and
        `UnassignedInt2`.
    header_cols : str or list of str or "all", optional
        Extra trace headers to load. Must be any of those specified in
        https://segyio.readthedocs.io/en/latest/segyio.html#trace-header-keys except for `UnassignedInt1` and
        `UnassignedInt2`.
        If not given, only headers from `header_index` are loaded and `TRACE_SEQUENCE_FILE` header is reconstructed
        automatically if not in the index.
        If "all", all available headers are loaded.
    name : str, optional
        Survey name. If not given, source file name is used. This name is mainly used to identify the survey when it is
        added to an index, see :class:`~index.SeismicIndex` docs for more info.
    limits : int or tuple or slice, optional
        Default time limits to be used during trace loading and survey statistics calculation. `int` or `tuple` are
        used as arguments to init a `slice` object. If not given, whole traces are used. Measured in samples.
    endian : {"big", "msb", "little", "lsb"}, optional, defaults to "big"
        SEG-Y file endianness.
    chunk_size : int, optional, defaults to 25000
        The number of traces to load by each of spawned processes.
    n_workers : int, optional
        The maximum number of simultaneously spawned processes to load trace headers. Defaults to the number of cpu
        cores.
    bar : bool, optional, defaults to True
        Whether to show survey loading progress bar.
    use_segyio_trace_loader : bool, optional, defaults to False
        Whether to use `segyio` trace loading methods or try optimizing data fetching using `numpy` memory mapping. May
        degrade performance if enabled.

    Attributes
    ----------
    path : str
        An absolute path to the source SEG-Y file.
    name : str
        Survey name.
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

    # pylint: disable-next=too-many-arguments, too-many-statements
    def __init__(self, path, header_index, header_cols=None, name=None, limits=None, endian="big", chunk_size=25000,
                 n_workers=None, bar=True, use_segyio_trace_loader=False):
        self.path = os.path.abspath(path)
        self.name = os.path.splitext(os.path.basename(self.path))[0] if name is None else name

        # Forbid loading UnassignedInt1 and UnassignedInt2 headers since they are treated differently from all other
        # headers by `segyio`
        allowed_headers = set(segyio.tracefield.keys.keys()) - {"UnassignedInt1", "UnassignedInt2"}

        header_index = to_list(header_index)
        if header_cols is None:
            header_cols = set()
        elif header_cols == "all":
            header_cols = allowed_headers
        else:
            header_cols = set(to_list(header_cols))

        # TRACE_SEQUENCE_FILE is not loaded but reconstructed manually since sometimes it is undefined in the file but
        # we rely on it during gather loading
        headers_to_load = (set(header_index) | header_cols) - {"TRACE_SEQUENCE_FILE"}

        unknown_headers = headers_to_load - allowed_headers
        if unknown_headers:
            raise ValueError(f"Unknown headers {', '.join(unknown_headers)}")

        # Open the SEG-Y file and memory map it
        if endian not in ENDIANNESS:
            raise ValueError(f"Unknown endian, must be one of {', '.join(ENDIANNESS)}")
        self.endian = endian
        self.segy_handler = segyio.open(self.path, mode="r", endian=endian, ignore_geometry=True)
        self.segy_handler.mmap()

        # Get attributes from the source SEG-Y file.
        self.file_sample_rate = self._infer_sample_rate()
        self.file_samples = (np.arange(self.segy_handler.trace.shape) * self.file_sample_rate).astype(np.float32)

        # Set samples and sample_rate according to passed `limits`.
        self.limits = None
        self.samples = None
        self.sample_rate = None
        self.set_limits(limits)

        # Load trace headers
        file_metrics = self.segy_handler.xfd.metrics()
        self.segy_format = file_metrics["format"]
        self.trace_data_offset = file_metrics["trace0"]
        self.n_file_traces = file_metrics["tracecount"]
        headers = load_headers(path, headers_to_load, trace_data_offset=self.trace_data_offset,
                               trace_size=file_metrics["trace_bsize"], n_traces=self.n_file_traces,
                               endian=endian, chunk_size=chunk_size, n_workers=n_workers, bar=bar)

        # Reconstruct TRACE_SEQUENCE_FILE header
        tsf_dtype = np.int32 if len(headers) < np.iinfo(np.int32).max else np.int64
        headers["TRACE_SEQUENCE_FILE"] = np.arange(1, self.segy_handler.tracecount+1, dtype=tsf_dtype)

        # Sort headers by the required index in order to optimize further subsampling and merging. Sorting preserves
        # trace order from the file within each gather.
        headers.set_index(header_index, inplace=True)
        headers.sort_index(kind="stable", inplace=True)

        # Set loaded survey headers and construct its fast indexer
        self._headers = None
        self._indexer = None
        self.headers = headers

        # Data format code defined by bytes 3225–3226 of the binary header that can be conveniently loaded using numpy
        # memmap. Currently only 3-byte integers (codes 7 and 15) and 4-byte fixed-point floats (code 4) are not
        # supported and result in a fallback to loading using segyio.
        endian_str = ">" if self.endian in {"big", "msb"} else "<"
        format_to_mmap_dtype = {
            1: np.uint8,  # IBM 4-byte float: read as 4 bytes and then manually transformed to an IEEE float32
            2: endian_str + "i4",
            3: endian_str + "i2",
            5: endian_str + "f4",
            6: endian_str + "f8",
            8: np.int8,
            9: endian_str + "i8",
            10: endian_str + "u4",
            11: endian_str + "u2",
            12: endian_str + "u8",
            16: np.uint8,
        }

        # Optionally create a memory map over traces data
        self.trace_dtype = self.segy_handler.dtype  # Appropriate data type of a buffer to load a trace into
        self.segy_trace_dtype = format_to_mmap_dtype.get(self.segy_format)  # Physical data type of traces on disc
        self.use_segyio_trace_loader = use_segyio_trace_loader or self.segy_trace_dtype is None
        self.traces_mmap = self._construct_traces_mmap()

        # Define all stats-related attributes
        self.has_stats = False
        self.min = None
        self.max = None
        self.mean = None
        self.std = None
        self.quantile_interpolator = None
        self.n_dead_traces = None

        # Define all geometry-related attributes and automatically infer field geometry if required headers are loaded
        self.has_inferred_geometry = False
        self._bins_to_coords_reg = None
        self._coords_to_bins_reg = None
        self.n_bins = None
        self.is_stacked = None
        self.bin_size = None  # m
        self.inline_length = None  # m
        self.crossline_length = None  # m
        self.area = None  # m^2
        self.is_2d = None
        if {"INLINE_3D", "CROSSLINE_3D", "CDP_X", "CDP_Y"} <= headers_to_load:
            self.infer_geometry()

    def _infer_sample_rate(self):
        """Get sample rate from file headers."""
        bin_sample_rate = self.segy_handler.bin[segyio.BinField.Interval]
        trace_sample_rate = self.segy_handler.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
        # 0 means undefined sample rate, so it is removed from the set of sample rate values.
        union_sample_rate = {bin_sample_rate, trace_sample_rate} - {0}
        if len(union_sample_rate) != 1:
            raise ValueError("Cannot infer sample rate from file headers: either both `Interval` (bytes 3217-3218 in "
                             "the binary header) and `TRACE_SAMPLE_INTERVAL` (bytes 117-118 in the header of the "
                             "first trace are undefined or they have different values.")
        return union_sample_rate.pop() / 1000  # Convert sample rate from microseconds to milliseconds

    def _construct_traces_mmap(self):
        """Memory map traces data."""
        if self.use_segyio_trace_loader:
            return None
        trace_shape = self.n_file_samples if self.segy_format != 1 else (self.n_file_samples, 4)
        mmap_trace_dtype = np.dtype([("headers", np.uint8, 240), ("trace", self.segy_trace_dtype, trace_shape)])
        return np.memmap(filename=self.path, mode="r", shape=self.n_file_traces, dtype=mmap_trace_dtype,
                         offset=self.trace_data_offset)["trace"]

    @property
    def n_file_samples(self):
        """int: Trace length in samples in the source SEG-Y file."""
        return len(self.file_samples)

    @property
    def dead_traces_marked(self):
        """bool: `mark_dead_traces` called."""
        return self.n_dead_traces is not None

    def __del__(self):
        """Close SEG-Y file handler on survey destruction."""
        self.segy_handler.close()

    def __getstate__(self):
        """Create a survey's pickling state from its `__dict__` by setting SEG-Y file handler and memory mapped trace
        data to `None`."""
        state = copy(self.__dict__)
        state["segy_handler"] = None
        state["traces_mmap"] = None
        return state

    def __setstate__(self, state):
        """Recreate a survey from unpickled state, reopen its source SEG-Y file and reconstruct a memory map over
        traces data."""
        self.__dict__ = state
        self.segy_handler = segyio.open(self.path, ignore_geometry=True)
        self.segy_handler.mmap()
        self.traces_mmap = self._construct_traces_mmap()

    def __str__(self):
        """Print survey metadata including information about source file and trace statistics if they were
        calculated."""
        offsets = self.headers.get('offset')
        offset_range = f'[{np.min(offsets)} m, {np.max(offsets)} m]' if offsets is not None else None
        msg = f"""
        Survey path:               {self.path}
        Survey name:               {self.name}
        Survey size:               {os.path.getsize(self.path) / (1024**3):4.3f} GB

        Indexed by:                {', '.join(to_list(self.indexed_by))}
        Number of gathers:         {self.n_gathers}
        Number of traces:          {self.n_traces}
        Trace length:              {self.n_samples} samples
        Sample rate:               {self.sample_rate} ms
        Times range:               [{min(self.samples)} ms, {max(self.samples)} ms]
        Offsets range:             {offset_range}
        """

        if self.has_inferred_geometry:
            msg += f"""
        Field geometry:
        Dimensionality:            {"2D" if self.is_2d else "3D"}
        Is stacked:                {self.is_stacked}
        Number of bins:            {self.n_bins}
        Area:                      {(self.area / 1000**2):.2f} km^2
        Bin size along inline:     {self.bin_size[0]:.1f} m
        Length along inline:       {(self.inline_length / 1000):.2f} km
        Bin size along crossline:  {self.bin_size[1]:.1f} m
        Length along crossline:    {(self.crossline_length / 1000):.2f} km
        """

        if self.has_stats:
            msg += f"""
        Survey statistics:
        mean | std:                {self.mean:>10.2f} | {self.std:<10.2f}
         min | max:                {self.min:>10.2f} | {self.max:<10.2f}
         q01 | q99:                {self.get_quantile(0.01):>10.2f} | {self.get_quantile(0.99):<10.2f}
        """

        if self.dead_traces_marked:
            msg += f"""
        Number of dead traces:     {self.n_dead_traces}
        """
        return dedent(msg).strip()

    def info(self):
        """Print survey metadata including information about source file and trace statistics if they were
        calculated."""
        print(self)

    #------------------------------------------------------------------------#
    #                     Statistics computation methods                     #
    #------------------------------------------------------------------------#

    def infer_geometry(self):
        coords_cols = ["CDP_X", "CDP_Y"]
        bins_cols = ["INLINE_3D", "CROSSLINE_3D"]
        cols = coords_cols + bins_cols

        # Construct a mapping from bins to their coordinates and back
        bins_to_coords = pd.DataFrame(self[cols], columns=cols)
        bins_to_coords = bins_to_coords.groupby(bins_cols, sort=False, as_index=False).agg("mean")
        bins_to_coords_reg = LinearRegression(copy_X=False, n_jobs=-1)
        bins_to_coords_reg.fit(bins_to_coords[bins_cols], bins_to_coords[coords_cols])
        coords_to_bins_reg = LinearRegression(copy_X=False, n_jobs=-1)
        coords_to_bins_reg.fit(bins_to_coords[coords_cols], bins_to_coords[bins_cols])

        # Set all geometry-related attributes
        self.has_inferred_geometry = True
        self._bins_to_coords_reg = bins_to_coords_reg
        self._coords_to_bins_reg = coords_to_bins_reg
        self.n_bins = len(bins_to_coords)
        self.is_stacked = (self.n_traces == self.n_bins)
        self.bin_size = np.diag(sp.linalg.polar(bins_to_coords_reg.coef_)[1])
        self.inline_length = (np.ptp(bins_to_coords["INLINE_3D"]) + 1) * self.bin_size[0]
        self.crossline_length = (np.ptp(bins_to_coords["CROSSLINE_3D"]) + 1) * self.bin_size[1]
        self.area = self.n_bins * np.prod(self.bin_size)
        self.is_2d = np.isclose(self.area, 0)
        return self

    @staticmethod
    def _cast_coords(coords, transformer):
        coords = np.array(coords)
        is_coords_1d = (coords.ndim == 1)
        coords = np.atleast_2d(coords)
        transformed_coords = transformer.predict(coords)
        if is_coords_1d:
            return transformed_coords[0]
        return transformed_coords

    def coords_to_bins(self, coords):
        if not self.has_inferred_geometry:
            raise ValueError("Survey geometry was not inferred, call `infer_geometry` method first.")
        return self._cast_coords(coords, self._coords_to_bins_reg)

    def bins_to_coords(self, bins):
        if not self.has_inferred_geometry:
            raise ValueError("Survey geometry was not inferred, call `infer_geometry` method first.")
        return self._cast_coords(bins, self._bins_to_coords_reg)

    # pylint: disable-next=too-many-statements
    def collect_stats(self, indices=None, n_quantile_traces=100000, quantile_precision=2, limits=None,
                      chunk_size=10000, bar=True):
        """Collect the following statistics by iterating over survey traces:
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
        indices : pd.Index, optional
            A subset of survey headers indices to collect stats for. If not given, statistics are calculated for the
            whole survey.
        n_quantile_traces : positive int, optional, defaults to 100000
            The number of traces to use for quantiles estimation.
        quantile_precision : positive int, optional, defaults to 2
            Calculate an approximate quantile for each q with `quantile_precision` decimal places. All other quantiles
            will be linearly interpolated on request.
        limits : int or tuple or slice, optional
            Time limits to be used for statistics calculation. `int` or `tuple` are used as arguments to init a `slice`
            object. If not given, `limits` passed to `__init__` are used. Measured in samples.
        chunk_size : int, optional, defaults to 10000
            The number of traces to be processed at once.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        survey : Survey
            The survey with collected stats. Sets `has_stats` flag to `True` and updates statistics attributes inplace.
        """
        if not self.dead_traces_marked or self.n_dead_traces:
            warnings.warn("The survey was not checked for dead traces or they were not removed. "
                          "Run `remove_dead_traces` first.", RuntimeWarning)

        limits = self.limits if limits is None else self._process_limits(limits)
        headers = self.headers
        if indices is not None:
            headers = self.get_headers_by_indices(indices)
        n_traces = len(headers)

        if n_quantile_traces < 0:
            raise ValueError("n_quantile_traces must be non-negative")
        # Clip n_quantile_traces if it's greater than the total number of traces
        n_quantile_traces = min(n_traces, n_quantile_traces)

        # Sort traces by TRACE_SEQUENCE_FILE: sequential access to trace amplitudes is much faster than random
        traces_pos = np.sort(get_cols(headers, "TRACE_SEQUENCE_FILE").ravel() - 1)
        quantile_traces_mask = np.zeros(n_traces, dtype=np.bool_)
        quantile_traces_mask[np.random.choice(n_traces, size=n_quantile_traces, replace=False)] = True

        # Split traces by chunks
        n_chunks, last_chunk_size = divmod(n_traces, chunk_size)
        chunk_sizes = [chunk_size] * n_chunks
        if last_chunk_size:
            n_chunks += 1
            chunk_sizes += [last_chunk_size]
        chunk_borders = np.cumsum(chunk_sizes[:-1])
        chunk_traces_pos = np.split(traces_pos, chunk_borders)
        chunk_quantile_traces_mask = np.split(quantile_traces_mask, chunk_borders)

        # Define buffers. chunk_mean, chunk_var and chunk_weights have float64 dtype to be numerically stable
        quantile_traces_buffer = []
        global_min, global_max = np.float32("inf"), np.float32("-inf")
        mean_buffer = np.empty(n_chunks, dtype=np.float64)
        var_buffer = np.empty(n_chunks, dtype=np.float64)
        chunk_weights = np.array(chunk_sizes, dtype=np.float64) / n_traces

        # Accumulate min, max, mean and var values of traces chunks
        bar_desc = f"Calculating statistics for traces in survey {self.name}"
        with tqdm(total=n_traces, desc=bar_desc, disable=not bar) as pbar:
            for i, (chunk_pos, chunk_quantile_mask) in enumerate(zip(chunk_traces_pos, chunk_quantile_traces_mask)):
                chunk_traces = self.load_traces(chunk_pos, limits=limits)
                if chunk_quantile_mask.any():
                    quantile_traces_buffer.append(chunk_traces[chunk_quantile_mask].ravel())

                chunk_min, chunk_max, chunk_mean, chunk_var = calculate_trace_stats(chunk_traces.ravel())
                global_min = min(chunk_min, global_min)
                global_max = max(chunk_max, global_max)
                mean_buffer[i] = chunk_mean
                var_buffer[i] = chunk_var
                pbar.update(len(chunk_traces))

        # Calculate global survey mean and variance by its values in chunks
        global_mean = np.average(mean_buffer, weights=chunk_weights)
        global_var = np.average(var_buffer + (mean_buffer - global_mean)**2, weights=chunk_weights)

        # Cast all calculated statistics to float32
        self.min = np.float32(global_min)
        self.max = np.float32(global_max)
        self.mean = np.float32(global_mean)
        self.std = np.float32(np.sqrt(global_var))

        if n_quantile_traces == 0:
            q = [0, 1]
            quantiles = [self.min, self.max]
        else:
            # Calculate all q-quantiles from 0 to 1 with step 1 / 10**quantile_precision
            q = np.round(np.linspace(0, 1, num=10**quantile_precision), decimals=quantile_precision)
            quantiles = np.nanquantile(np.concatenate(quantile_traces_buffer), q=q)
            # 0 and 1 quantiles are replaced with actual min and max values respectively
            quantiles[0], quantiles[-1] = self.min, self.max
        self.quantile_interpolator = interp1d(q, quantiles)

        self.has_stats = True
        return self

    def mark_dead_traces(self, limits=None, bar=True):
        """Mark dead traces (those having constant amplitudes) by setting a value of a new `DeadTrace`
        header to `True` and store the overall number of dead traces in the `n_dead_traces` attribute.

        Parameters
        ----------
        limits : int or tuple or slice, optional
            Time limits to be used to detect dead traces. `int` or `tuple` are used as arguments to init a `slice`
            object. If not given, `limits` passed to `__init__` are used. Measured in samples.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        survey : Survey
            The same survey with a new `DeadTrace` header created.
        """

        limits = self.limits if limits is None else self._process_limits(limits)

        traces_pos = self["TRACE_SEQUENCE_FILE"].ravel() - 1
        n_samples = len(self.file_samples[limits])

        trace = np.empty(n_samples, dtype=self.trace_dtype)
        dead_indices = []
        for tr_index, pos in tqdm(enumerate(traces_pos), desc=f"Detecting dead traces for survey {self.name}",
                                  total=len(self.headers), disable=not bar):
            self.load_trace_segyio(buf=trace, index=pos, limits=limits, trace_length=n_samples)
            trace_min, trace_max, *_ = calculate_trace_stats(trace)

            if math.isclose(trace_min, trace_max):
                dead_indices.append(tr_index)

        self.n_dead_traces = len(dead_indices)
        self.headers[HDR_DEAD_TRACE] = False
        self.headers.iloc[dead_indices, self.headers.columns.get_loc(HDR_DEAD_TRACE)] = True

        return self

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

    def load_trace_segyio(self, buf, index, limits, trace_length):
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
        buf : 1d np.ndarray of self.trace_dtype
            An empty array to save the loaded trace.
        index : int
            Trace position in the file.
        limits : slice
            Trace time range to load. Measured in samples.
        trace_length : int
            Total number of samples to load.

        Returns
        -------
        trace : 1d np.ndarray of self.trace_dtype
            Loaded trace.
        """
        return self.segy_handler.xfd.gettr(buf, index, 1, 1, limits.start, limits.stop, limits.step, trace_length)

    def load_traces_segyio(self, traces_pos, limits=None):
        """Load traces by their positions in the SEG-Y file using low-level `segyio` interface."""
        limits = self.limits if limits is None else self._process_limits(limits)
        samples = self.file_samples[limits]
        n_samples = len(samples)

        traces = np.empty((len(traces_pos), n_samples), dtype=self.trace_dtype)
        for i, pos in enumerate(traces_pos):
            self.load_trace_segyio(buf=traces[i], index=pos, limits=limits, trace_length=n_samples)
        return traces

    def load_traces_mmap(self, traces_pos, limits=None):
        """Load traces by their positions in the SEG-Y file from memory mapped trace data."""
        limits = self.limits if limits is None else self._process_limits(limits)
        if self.segy_format != 1:
            return self.traces_mmap[traces_pos, limits]
        # IBM 4-byte float case: reading from mmap with step is way more expensive
        # than loading the whole trace with consequent slicing
        traces = self.traces_mmap[traces_pos, limits.start:limits.stop]
        if limits.step != 1:
            traces = traces[:, ::limits.step]
        traces_bytes = (traces[:, :, 0], traces[:, :, 1], traces[:, :, 2], traces[:, :, 3])
        if self.endian in {"little", "lsb"}:
            traces_bytes = traces_bytes[::-1]
        return ibm_to_ieee(*traces_bytes)

    def load_traces(self, traces_pos, limits=None):
        """Load traces by their positions in the SEG-Y file."""
        loader = self.load_traces_segyio if self.use_segyio_trace_loader else self.load_traces_mmap
        traces = loader(traces_pos, limits=limits)
        # Cast the result to a C-contiguous float32 array regardless of the dtype in the source file
        return np.require(traces, dtype=np.float32, requirements="C")

    def load_gather(self, headers, limits=None, copy_headers=False):
        """Load a gather with given `headers`.

        Parameters
        ----------
        headers : pd.DataFrame
            Headers of traces to load. Must be a subset of `self.headers`.
        limits : int or tuple or slice or None, optional
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, `limits` passed to `__init__` are used. Measured in samples.
        copy_headers : bool, optional, defaults to False
            Whether to copy the passed `headers` when instantiating the gather.

        Returns
        -------
        gather : Gather
            Loaded gather instance.
        """
        if copy_headers:
            headers = headers.copy()
        traces_pos = get_cols(headers, "TRACE_SEQUENCE_FILE").ravel() - 1
        limits = self.limits if limits is None else self._process_limits(limits)
        samples = self.file_samples[limits]
        data = self.load_traces(traces_pos, limits=limits)
        return Gather(headers=headers, data=data, samples=samples, survey=self)

    def get_gather(self, index, limits=None, copy_headers=False):
        """Load a gather with given `index`.

        Parameters
        ----------
        index : int or 1d array-like
            An index of the gather to load. Must be one of `self.indices`.
        limits : int or tuple or slice or None, optional
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, `limits` passed to `__init__` are used. Measured in samples.
        copy_headers : bool, optional, defaults to False
            Whether to copy the subset of survey `headers` describing the gather.

        Returns
        -------
        gather : Gather
            Loaded gather instance.
        """
        return self.load_gather(self.get_headers_by_indices((index,)), limits=limits, copy_headers=copy_headers)

    def sample_gather(self, limits=None, copy_headers=False):
        """Load a gather with random index.

        Parameters
        ----------
        limits : int or tuple or slice or None, optional
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, `limits` passed to `__init__` are used. Measured in samples.
        copy_headers : bool, optional, defaults to False
            Whether to copy the subset of survey `headers` describing the sampled gather.

        Returns
        -------
        gather : Gather
            Loaded gather instance.
        """
        return self.get_gather(index=np.random.choice(self.indices), limits=limits, copy_headers=copy_headers)

    # pylint: disable=anomalous-backslash-in-string
    def load_first_breaks(self, path, trace_id_cols=('FieldRecord', 'TraceNumber'), first_breaks_col=HDR_FIRST_BREAK,
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
        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment

        # if decimal is not provided, try to infer it from the first line
        if decimal is None:
            with open(path, 'r', encoding=encoding) as f:
                row = f.readline()
            decimal = '.' if '.' in row else ','

        file_columns = to_list(trace_id_cols) + [first_breaks_col]
        first_breaks_df = pd.read_csv(path, delimiter=delimiter, names=file_columns,
                                      decimal=decimal, encoding=encoding, **kwargs)

        headers = self.headers
        headers_index = self.indexed_by
        headers.reset_index(inplace=True)
        headers = headers.merge(first_breaks_df, on=trace_id_cols)
        if headers.empty:
            raise ValueError('Empty headers after first breaks loading.')
        headers.set_index(headers_index, inplace=True)
        headers.sort_index(kind="stable", inplace=True)
        self.headers = headers
        return self

    #------------------------------------------------------------------------#
    #                       Survey processing methods                        #
    #------------------------------------------------------------------------#

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

    def remove_dead_traces(self, limits=None, inplace=False, bar=True):
        """ Remove dead (constant) traces from the survey.
        Calls `mark_dead_traces` if it was not called before.

        Parameters
        ----------
        limits : int or tuple or slice, optional
            Time limits to be used to detect dead traces if needed. `int` or `tuple` are used as arguments to init a
            `slice` object. If not given, `limits` passed to `__init__` are used. Measured in samples.
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
        if not self.dead_traces_marked:
            self.mark_dead_traces(limits=limits, bar=bar)

        self.filter(lambda dt: ~dt, cols=HDR_DEAD_TRACE, inplace=True)
        self.n_dead_traces = 0
        return self

    #------------------------------------------------------------------------#
    #                         Task specific methods                          #
    #------------------------------------------------------------------------#

    def generate_supergathers(self, centers=None, origin=None, size=3, step=20, border_indent=0, strict=True,
                              max_trials=10000, reindex=True, inplace=False):
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
        size = np.broadcast_to(size, 2)
        step = np.broadcast_to(step, 2)

        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        line_cols = ["INLINE_3D", "CROSSLINE_3D"]
        super_line_cols = ["SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"]
        index_cols = super_line_cols if reindex else self.indexed_by

        if centers is None:
            # Find unique pairs of inlines and crosslines, drop_duplicates is way faster than np.unique
            lines = pd.DataFrame(self[["INLINE_3D", "CROSSLINE_3D"]]).drop_duplicates().values

            # Construct a binary mask of a field where True value is set for bins containing at least one trace
            # and False otherwise
            lines_min = lines.min(axis=0)
            lines_normed = lines - lines_min
            contour_img = np.zeros(lines_normed.max(axis=0) + 1, dtype=np.uint8)
            contour_img[lines_normed[:, 0], lines_normed[:, 1]] = 1

            # Erode field mask according to border_indent and strict flag
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, np.broadcast_to(border_indent, 2) * 2 + 1)
            contour_img = cv2.erode(contour_img, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if strict:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
                contour_img = cv2.erode(contour_img, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            step = np.minimum(step, contour_img.shape)

            # Construct a regular grid of supergathers covering the field with a margin of step
            supergather_mask = np.zeros(contour_img.shape + step - 1, dtype=np.uint8)
            i, x = np.meshgrid(np.arange(0, supergather_mask.shape[0], step[0]),
                               np.arange(0, supergather_mask.shape[1], step[1]))
            supergather_mask[i, x] = 1

            if origin is not None:
                origin = np.broadcast_to(origin, 2)
                best_shift = (lines_min - origin) % step
            else:
                # Calculate the number of shifts along inline and crossline axes
                n_i_shifts = np.clip(int(np.sqrt(max_trials * step[0] / step[1])), 1, min(step[0], max_trials))
                n_x_shifts = max_trials // n_i_shifts
                if n_x_shifts > step[1]:
                    n_x_shifts = step[1]
                    n_i_shifts = min(max_trials // n_x_shifts, step[0])

                # Calculate the number of supergathers for each shift
                n_supergathers = {}
                for i_shift in int_linspace(0, step[0] - 1, n_i_shifts):
                    for x_shift in int_linspace(0, step[1] - 1, n_x_shifts):
                        shifted_mask = supergather_mask[i_shift : i_shift + contour_img.shape[0],
                                                        x_shift : x_shift + contour_img.shape[1]]
                        n_supergathers[(i_shift, x_shift)] = (contour_img & shifted_mask).sum()

                # Select all shifts producing maximum number of supergathers
                max_supergathers = max(n_supergathers.values())
                optimal_shifts = np.stack([shift for shift, n in n_supergathers.items() if n == max_supergathers])
                optimal_masks = np.stack([supergather_mask[i_shift : i_shift + contour_img.shape[0],
                                                        x_shift : x_shift + contour_img.shape[1]]
                                        for i_shift, x_shift in optimal_shifts])

                # Select a shift whose supergathers are further away from field borders
                eroded_img = contour_img
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                for _ in range(size.max()):
                    eroded_img = cv2.erode(eroded_img, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
                    drop_mask = (~eroded_img & optimal_masks).any(axis=(1, 2))
                    if (len(drop_mask) == 1) or drop_mask.all():
                        break
                    optimal_shifts = optimal_shifts[~drop_mask]
                    optimal_masks = optimal_masks[~drop_mask]
                best_shift = optimal_shifts[0]

            # Generate supergather centers
            best_mask = supergather_mask[best_shift[0] : best_shift[0] + contour_img.shape[0],
                                         best_shift[1] : best_shift[1] + contour_img.shape[1]]
            centers = np.nonzero(contour_img & best_mask)
            centers = np.stack(centers).T + lines_min

        centers = np.array(centers)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("Passed centers must have shape (n_supergathers, 2)")

        # Construct a bridge table with mapping from supergather centers to their bins
        shifts_grid = np.meshgrid(np.arange(size[0]) - size[0] // 2, np.arange(size[1]) - size[1] // 2)
        shifts = np.stack(shifts_grid).reshape(2, -1).T
        bridge = np.column_stack([centers.repeat(size.prod(), axis=0), (centers[:, None] + shifts).reshape(-1, 2)])
        bridge = pd.DataFrame(bridge, columns=super_line_cols+line_cols)

        headers = self.headers
        headers.reset_index(inplace=True)
        headers = pd.merge(bridge, headers, on=line_cols)
        headers.set_index(index_cols, inplace=True)
        headers.sort_index(kind="stable", inplace=True)
        self.headers = headers
        return self

    #------------------------------------------------------------------------#
    #                         Visualization methods                          #
    #------------------------------------------------------------------------#

    def plot_geometry(self, **kwargs):
        """Plot shot and receiver locations on a field map.

        This plot is interactive and provides 2 views:
        * Shot view: displays shot locations. Highlights all activated receivers on click and displays the
          corresponding common shot gather.
        * Receiver view: displays receiver locations. Highlights all shots that activated the receiver on click and
          displays the corresponding common receiver gather.

        Plotting must be performed in a JupyterLab environment with the the `%matplotlib widget` magic executed and
        `ipympl` and `ipywidgets` libraries installed.

        Parameters
        ----------
        keep_aspect : bool, optional, defaults to False
            Whether to keep aspect ratio of the map plot.
        x_ticker : str or dict, optional
            Parameters to control `x` axis tick formatting and layout of the map plot. See `.utils.set_ticks` for more
            details.
        y_ticker : dict, optional
            Parameters to control `y` axis tick formatting and layout of the map plot. See `.utils.set_ticks` for more
            details.
        sort_by : str, optional
            Header name to sort the displayed gather by.
        gather_plot_kwargs : dict, optional
            Additional arguments to pass to `Gather.plot`.
        figsize : tuple with 2 elements, optional, defaults to (4.5, 4.5)
            Size of created map and gather figures. Measured in inches.
        kwargs : misc, optional
            Additional keyword arguments to pass to `matplotlib.axes.Axes.scatter` when plotting the map.
        """
        SurveyGeometryPlot(self, **kwargs).plot()

    def construct_attribute_map(self, attribute, by, drop_duplicates=False, agg=None, bin_size=None, **kwargs):
        """Construct a map of trace attributes aggregated by gathers.

        Examples
        --------
        Construct a map of maximum offsets by shots:
        >>> max_offset_map = survey.construct_attribute_map("offset", by="shot", agg="max")
        >>> max_offset_map.plot()

        The map allows for interactive plotting: a gather type defined by `by` will be displayed on click on the map.
        The gather may be optionally sorted if `sort_by` argument if passed to the `plot` method:
        >>> max_offset_map.plot(interactive=True, sort_by="offset")

        Generate supergathers and calculate the number of traces in each of them (fold):
        >>> supergather_columns = ["SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"]
        >>> supergather_survey = survey.generate_supergathers(size=(7, 7), step=(7, 7))
        >>> fold_map = supergather_survey.construct_attribute_map("fold", by=supergather_columns)
        >>> fold_map.plot()

        Parameters
        ----------
        attribute : str
            If "fold", calculates the number of traces in gathers defined by `by`. Otherwise defines a survey header
            name to construct a map for.
        by : tuple with 2 elements or {"shot", "receiver", "midpoint", "bin"}
            If `tuple`, survey headers names to get coordinates from.
            If `str`, gather type to aggregate header values over.
        drop_duplicates : bool, optional, defaults to False
            Whether to drop duplicated (coordinates, value) pairs. Useful when dealing with an attribute defined for a
            shot or receiver, not a trace (e.g. constructing a map of elevations by shots).
        agg : str or callable, optional, defaults to "mean"
            An aggregation function. Passed directly to `pandas.core.groupby.DataFrameGroupBy.agg`.
        bin_size : int, float or array-like with length 2, optional
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.
        kwargs : misc, optional
            Additional keyword arguments to pass to `Metric.__init__`.

        Returns
        -------
        attribute_map : BaseMetricMap
            Constructed attribute map.
        """
        if isinstance(by, str):
            by_to_coords_cols = {
                "shot": ["SourceX", "SourceY"],
                "receiver": ["GroupX", "GroupY"],
                "midpoint": ["CDP_X", "CDP_Y"],
                "bin": ["INLINE_3D", "CROSSLINE_3D"],
            }
            coords_cols = by_to_coords_cols.get(by)
            if coords_cols is None:
                raise ValueError(f"by must be one of {', '.join(by_to_coords_cols.keys())} but {by} given.")
        else:
            coords_cols = to_list(by)
        if len(coords_cols) != 2:
            raise ValueError("Exactly 2 coordinates headers must be passed")

        if attribute == "fold":
            map_data = self.headers.groupby(coords_cols, as_index=False).size().rename(columns={"size": "Fold"})
        else:
            data_cols = coords_cols + [attribute]
            map_data = pd.DataFrame(self[data_cols], columns=data_cols)
            if drop_duplicates:
                map_data.drop_duplicates(inplace=True)

        metric = PartialMetric(SurveyAttribute, survey=self, name=attribute, **kwargs)
        return metric.map_class(map_data.iloc[:, :2], map_data.iloc[:, 2], metric=metric, agg=agg, bin_size=bin_size)
