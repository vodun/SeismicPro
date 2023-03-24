"""Implements Survey class describing a single SEG-Y file"""

import os
import warnings
from copy import copy
from textwrap import dedent
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import cv2
import segyio
import numpy as np
import scipy as sp
import pandas as pd
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from .headers import load_headers
from .headers_checks import validate_trace_headers, validate_source_headers, validate_receiver_headers
from .metrics import (SurveyAttribute, TracewiseMetric, BaseWindowMetric, MetricsRatio, DeadTrace,
                      DEFAULT_TRACEWISE_METRICS)
from .plot_geometry import SurveyGeometryPlot
from .utils import ibm_to_ieee, calculate_trace_stats
from ..gather import Gather
from ..containers import GatherContainer, SamplesContainer
from ..metrics import initialize_metrics
from ..utils import to_list, maybe_copy, get_cols, get_first_defined, get_cols_from_by
from ..const import ENDIANNESS, HDR_FIRST_BREAK, HDR_TRACE_POS


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
    pipelines. All loaded headers are stored in the `headers` attribute as a `pd.DataFrame` with `header_index` columns
    set as its index.

    Values of both `header_index` and `header_cols` must be any of those specified in
    https://segyio.readthedocs.io/en/latest/segyio.html#trace-header-keys except for `UnassignedInt1` and
    `UnassignedInt2` since they are treated differently from all other headers by `segyio`. Also, `TRACE_SEQUENCE_FILE`
    header is not loaded from the file but always automatically reconstructed.

    The survey sample rate is calculated by two values stored in:
    - bytes 3217-3218 of the binary header, called `Interval` in `segyio`,
    - bytes 117-118 of the trace header of the first trace in the file, called `TRACE_SAMPLE_INTERVAL` in `segyio`.
    If both of them are present and equal or only one of them is well-defined (non-zero), it is used as a sample rate.
    Otherwise, an error is raised.

    If `INLINE_3D` and `CROSSLINE_3D` trace headers are loaded, properties of survey binning are automatically inferred
    on survey construction which allows accessing:
    - Some bin-related attributes of the survey, such as `n_bins`,
    - `dist_to_bin_contours` method which calculates distances from points to a contour of the survey in bin
      coordinates.

    If `CDP_X` and `CDP_Y` headers are loaded together with `INLINE_3D` and `CROSSLINE_3D`, field geometry is also
    inferred which allows accessing:
    - Some geometry-related attributes of the survey, such as `area`, `perimeter` and `bin_size`,
    - `coords_to_bins` and `bins_to_coords` methods which convert geographic coordinates to bins and back respectively,
    - `dist_to_geographic_contours` method which calculates distances from points to a contour of the survey in
      geographic coordinates.

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
    source_id_cols : str or list of str, optional
        Trace headers that uniquely identify a seismic source. If not given, set in the following way (in order of
        priority):
        - `FieldRecord` if it loaded,
        - [`SourceX`, `SourceY`] if they are loaded,
        - `None` otherwise.
    receiver_id_cols : str or list of str, optional
        Trace headers that uniquely identify a receiver. If not given, set to [`GroupX`, `GroupY`] if they are loaded.
    name : str, optional
        Survey name. If not given, source file name is used. This name is mainly used to identify the survey when it is
        added to an index, see :class:`~index.SeismicIndex` docs for more info.
    limits : int or tuple or slice, optional
        Default time limits to be used during trace loading and survey statistics calculation. `int` or `tuple` are
        used as arguments to init a `slice` object. If not given, whole traces are used. Measured in samples.
    validate : bool, optional, defaults to True
        Whether to perform validation of trace headers consistency.
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
    source_id_cols : str or list of str or None
        Trace headers that uniquely identify a seismic source.
    receiver_id_cols : str or list of str or None
        Trace headers that uniquely identify a receiver.
    segy_handler : segyio.segy.SegyFile
        Source SEG-Y file handler.
    has_stats : bool
        Whether the survey has trace statistics calculated. `False` until `collect_stats` method is called.
    min : np.float32 or None
        Minimum trace value. `None` until trace statistics are calculated.
    max : np.float32 or None
        Maximum trace value. `None` until trace statistics are calculated.
    mean : np.float32 or None
        Mean trace value. `None` until trace statistics are calculated.
    std : np.float32 or None
        Standard deviation of trace values. `None` until trace statistics are calculated.
    quantile_interpolator : scipy.interpolate.interp1d or None
        Interpolator of trace values quantiles. `None` until trace statistics are calculated.
    n_dead_traces : int or None
        The number of traces with constant value (dead traces). `None` until `mark_dead_traces` method is called.
    has_inferred_binning : bool
        Whether properties of survey binning have been inferred. `True` if `INLINE_3D` and `CROSSLINE_3D` trace headers
        are loaded on survey instantiation or `infer_binning` method is explicitly called.
    n_bins : int or None
        The number of bins in the survey. `None` until properties of survey binning are inferred.
    is_stacked : bool or None
        Whether the survey is stacked. `None` until properties of survey binning are inferred.
    field_mask : 2d np.ndarray or None
        A binary mask of the field with ones set for bins with at least one trace and zeros otherwise. `None` until
        properties of survey binning are inferred.
    field_mask_origin : np.ndarray with 2 elements
        Minimum values of inline and crossline over the field. `None` until properties of survey binning are inferred.
    bin_contours : tuple of np.ndarray or None
        Contours of all connected components of the field in bin coordinates. `None` until properties of survey binning
        are inferred.
    has_inferred_geometry : bool
        Whether the survey has inferred geometry. `True` if `INLINE_3D`, `CROSSLINE_3D`, `CDP_X` and `CDP_Y` trace
        headers are loaded on survey instantiation or `infer_geometry` method is explicitly called.
    is_2d : bool or None
        Whether the survey is 2D. `None` until survey geometry is inferred.
    bin_size : np.ndarray with 2 elements or None
        Bin sizes in meters along inline and crossline directions. `None` until survey geometry is inferred.
    inline_length : float or None
        Maximum field length along inline direction in meters. `None` until survey geometry is inferred.
    crossline_length : float or None
        Maximum field length along crossline direction in meters. `None` until survey geometry is inferred.
    area : float or None
        Field area in squared meters. `None` until survey geometry is inferred.
    perimeter : float or None
        Field perimeter in meters. `None` until survey geometry is inferred.
    geographic_contours : tuple of np.ndarray or None
        Contours of all connected components of the field in geographic coordinates. `None` until survey geometry is
        inferred.
    """

    # pylint: disable-next=too-many-arguments, too-many-statements
    def __init__(self, path, header_index, header_cols=None, source_id_cols=None, receiver_id_cols=None, name=None,
                 limits=None, validate=True, endian="big", chunk_size=25000, n_workers=None, bar=True,
                 use_segyio_trace_loader=False):
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
        headers_to_load = set(header_index) | header_cols

        # Parse source and receiver id cols and set defaults if needed
        if source_id_cols is None:
            if "FieldRecord" in headers_to_load:
                source_id_cols = "FieldRecord"
            elif {"SourceX", "SourceY"} <= headers_to_load:
                source_id_cols = ["SourceX", "SourceY"]
        else:
            headers_to_load |= set(to_list(source_id_cols))
        self.source_id_cols = source_id_cols

        if receiver_id_cols is None:
            if {"GroupX", "GroupY"} <= headers_to_load:
                receiver_id_cols = ["GroupX", "GroupY"]
        else:
            headers_to_load |= set(to_list(receiver_id_cols))
        self.receiver_id_cols = receiver_id_cols

        # TRACE_SEQUENCE_FILE is not loaded but reconstructed manually since sometimes it is undefined in the file but
        # we rely on it during gather loading
        headers_to_load = headers_to_load - {"TRACE_SEQUENCE_FILE"}

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

        # Validate trace headers for consistency
        if validate:
            self.validate_headers()

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

        # calculated QC metrics
        self.qc_metrics = {}

        # Define all bin-related attributes and automatically infer them if both INLINE_3D and CROSSLINE_3D are loaded
        self.has_inferred_binning = False
        self.n_bins = None
        self.is_stacked = None
        self.field_mask = None
        self.field_mask_origin = None
        self.bin_contours = None
        if {"INLINE_3D", "CROSSLINE_3D"} <= headers_to_load:
            self.infer_binning()

        # Define all geometry-related attributes and automatically infer field geometry if required headers are loaded
        self.has_inferred_geometry = False
        self._bins_to_coords_reg = None
        self._coords_to_bins_reg = None
        self.is_2d = None
        self.area = None  # m^2
        self.perimeter = None  # m
        self.bin_size = None  # (m, m)
        self.inline_length = None  # m
        self.crossline_length = None  # m
        self.geographic_contours = None
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
    def n_sources(self):
        """int: The number of sources."""
        if self.source_id_cols is None:
            return None
        return len(self.get_headers(self.source_id_cols).drop_duplicates())

    @property
    def n_receivers(self):
        """int: The number of receivers."""
        if self.receiver_id_cols is None:
            return None
        return len(self.get_headers(self.receiver_id_cols).drop_duplicates())

    @property
    def is_uphole(self):
        """bool or None: Whether the survey is uphole. `None` if uphole-related headers are not loaded."""
        has_uphole_times = "SourceUpholeTime" in self.available_headers
        has_uphole_depths = "SourceDepth" in self.available_headers
        has_positive_uphole_times = has_uphole_times and (self["SourceUpholeTime"] > 0).any()
        has_positive_uphole_depths = has_uphole_depths and (self["SourceDepth"] > 0).any()
        if not has_uphole_times and not has_uphole_depths:
            return None
        return has_positive_uphole_times or has_positive_uphole_depths

    @GatherContainer.headers.setter
    def headers(self, headers):
        """Reconstruct trace positions on each headers assignment."""
        GatherContainer.headers.fset(self, headers)
        htp_dtype = np.int32 if len(headers) < np.iinfo(np.int32).max else np.int64
        self.headers[HDR_TRACE_POS] = np.arange(self.n_traces, dtype=htp_dtype)

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
        """Print survey metadata including information about the source file, field geometry if it was inferred and
        trace statistics if they were calculated."""
        offsets = self.headers.get('offset')
        offset_range = f"[{np.min(offsets)} m, {np.max(offsets)} m]" if offsets is not None else "Unknown"

        msg = f"""
        Survey path:               {self.path}
        Survey name:               {self.name}
        Survey size:               {os.path.getsize(self.path) / (1024**3):4.3f} GB

        Number of traces:          {self.n_traces}
        Trace length:              {self.n_samples} samples
        Sample rate:               {self.sample_rate} ms
        Times range:               [{min(self.samples)} ms, {max(self.samples)} ms]
        Offsets range:             {offset_range}
        Is uphole:                 {get_first_defined(self.is_uphole, "Unknown")}

        Indexed by:                {", ".join(to_list(self.indexed_by))}
        Number of gathers:         {self.n_gathers}
        Mean gather fold:          {int(self.n_traces / self.n_gathers)}
        """

        if self.has_inferred_binning:
            msg += f"""
        Is stacked:                {self.is_stacked}
        Number of bins:            {self.n_bins}
        Mean bin fold:             {int(self.n_traces / self.n_bins)}
        """

        if self.source_id_cols is not None:
            n_sources = self.n_sources  # Run possibly time-consuming calculation once
            msg += f"""
        Source ID headers:         {", ".join(to_list(self.source_id_cols))}
        Number of sources:         {n_sources}
        Mean source fold:          {int(self.n_traces / n_sources)}
        """

        if self.receiver_id_cols is not None:
            n_receivers = self.n_receivers  # Run possibly time-consuming calculation once
            msg += f"""
        Receiver ID headers:       {", ".join(to_list(self.receiver_id_cols))}
        Number of receivers:       {n_receivers}
        Mean receiver fold:        {int(self.n_traces / n_receivers)}
        """

        if self.has_inferred_geometry:
            msg += f"""
        Field geometry:
        Dimensionality:            {"2D" if self.is_2d else "3D"}
        Area:                      {(self.area / 1000**2):.2f} km^2
        Perimeter:                 {(self.perimeter / 1000):.2f} km
        Inline bin size:           {self.bin_size[0]:.1f} m
        Crossline bin size:        {self.bin_size[1]:.1f} m
        Inline length:             {(self.inline_length / 1000):.2f} km
        Crossline length:          {(self.crossline_length / 1000):.2f} km
        """

        if self.has_stats:
            msg += f"""
        Survey statistics:
        mean | std:                {self.mean:>10.2f} | {self.std:<10.2f}
         min | max:                {self.min:>10.2f} | {self.max:<10.2f}
         q01 | q99:                {self.get_quantile(0.01):>10.2f} | {self.get_quantile(0.99):<10.2f}
        """

        if self.qc_metrics:
            metric_msg = ""
            for metric in self.qc_metrics.values():
                if metric.threshold is None:
                    continue
                metric_value = self.headers[metric.header_cols]
                if isinstance(metric, BaseWindowMetric):
                    metric_value = metric.compute_rms(*self[metric.header_cols].T)
                metric_msg += f"\n\t{metric.description+':':<50}{sum(metric.binarize(metric_value))}"
            if metric_msg:
                msg += """
        Number of bad traces after tracewise QC found by:
        """ + metric_msg
        return dedent(msg).strip()

    def info(self):
        """Print survey metadata including information about the source file, field geometry if it was inferred and
        trace statistics if they were calculated."""
        print(self)

    def set_source_id_cols(self, cols, validate=True):
        """Set new trace headers that uniquely identify a seismic source and optionally validate consistency of
        source-related trace headers by checking that each source has unique coordinates, surface elevation, uphole
        time and depth."""
        if set(to_list(cols)) - self.available_headers:
            raise ValueError("Required headers were not loaded")
        if validate:
            headers = self.headers.copy(deep=False)
            headers.reset_index(inplace=True)
            validate_source_headers(headers, cols)
        self.source_id_cols = cols

    def set_receiver_id_cols(self, cols, validate=True):
        """Set new trace headers that uniquely identify a receiver and optionally validate consistency of
        receiver-related trace headers by checking that each receiver has unique coordinates and surface elevation."""
        if set(to_list(cols)) - self.available_headers:
            raise ValueError("Required headers were not loaded")
        if validate:
            headers = self.headers.copy(deep=False)
            headers.reset_index(inplace=True)
            validate_receiver_headers(headers, cols)
        self.receiver_id_cols = cols

    def validate_headers(self, offset_atol=10, cdp_atol=10, elevation_atol=5, elevation_radius=50):
        """Check trace headers for consistency.

        1. Validate trace headers by checking that:
           - All headers are not empty,
           - Trace identifier (FieldRecord, TraceNumber) has no duplicates,
           - Source uphole times and depths are non-negative,
           - Source uphole time is zero if and only if source depth is also zero,
           - Traces do not have signed offsets,
           - Offsets in trace headers coincide with distances between sources (SourceX, SourceY) and receivers (GroupX,
             GroupY),
           - Coordinates of a midpoint (CDP_X, CDP_Y) matches those of the corresponding source (SourceX, SourceY) and
             receiver (GroupX, GroupY),
           - Surface elevation is unique for a given spatial location,
           - Elevation-related headers (ReceiverGroupElevation, SourceSurfaceElevation) have consistent ranges,
           - Mapping from geographic (CDP_X, CDP_Y) to line-based (INLINE_3D, CROSSLINE_3D) coordinates and back is
             unique.

        2. Validate consistency of source-related trace headers by checking that each source has unique coordinates,
           surface elevation, uphole time and depth.

        3. Validate consistency of receiver-related trace headers by checking that each receiver has unique coordinates
           and surface elevation.

        If any of the checks fail, a warning is displayed.

        Parameters
        ----------
        offset_atol : int, optional, defaults to 10
            Maximum allowed difference between a trace offset and the distance between its source and receiver.
        cdp_atol : int, optional, defaults to 10
            Maximum allowed difference between coordinates of a trace CDP and the midpoint between its source and
            receiver.
        elevation_atol : int, optional, defaults to 5
            Maximum allowed difference between surface elevation at a given source/receiver location and mean elevation
            of all sources and receivers within a radius defined by `elevation_radius`.
        elevation_radius : int, optional, defaults to 50
            Radius of the neighborhood to estimate mean surface elevation.
        """
        headers = self.headers.copy(deep=False)
        headers.reset_index(inplace=True)
        validate_trace_headers(headers, offset_atol=offset_atol, cdp_atol=cdp_atol, elevation_atol=elevation_atol,
                               elevation_radius=elevation_radius)
        validate_source_headers(headers, self.source_id_cols)
        validate_receiver_headers(headers, self.receiver_id_cols)

    #------------------------------------------------------------------------#
    #                        Geometry-related methods                        #
    #------------------------------------------------------------------------#

    def infer_binning(self):
        """Infer properties of survey binning by estimating the following entities:
        1. Number of bins,
        2. Pre- or post-stack flag,
        3. Binary mask of the field and its origin,
        4. Field contours in bin coordinate system.

        After the method is executed `has_inferred_binning` flag is set to `True` and all the calculated values can be
        obtained via corresponding attributes.
        """
        # Find unique pairs of inlines and crosslines, drop_duplicates is way faster than np.unique
        lines = self.get_headers(["INLINE_3D", "CROSSLINE_3D"]).drop_duplicates().to_numpy()

        # Construct a binary mask of a field where True value is set for bins containing at least one trace
        # and False otherwise
        origin = lines.min(axis=0)
        normed_lines = lines - origin
        field_mask = np.zeros(normed_lines.max(axis=0) + 1, dtype=np.uint8)
        field_mask[normed_lines[:, 0], normed_lines[:, 1]] = 1
        bin_contours = cv2.findContours(field_mask.T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=origin)[0]

        # Set all bin-related attributes
        self.has_inferred_binning = True
        self.n_bins = len(lines)
        self.is_stacked = self.n_traces == self.n_bins
        self.field_mask = field_mask
        self.field_mask_origin = origin
        self.bin_contours = bin_contours

    def infer_geometry(self):
        """Infer survey geometry by estimating the following entities:
        1. Survey dimensionality (2D/3D),
        2. Bin sizes along inline and crossline directions,
        3. Survey lengths along inline and crossline directions,
        4. Survey area and perimeter,
        5. Field contours in geographic coordinate system,
        6. Mappings from geographic coordinates to bins and back.

        After the method is executed `has_inferred_geometry` flag is set to `True` and all the calculated values can be
        obtained via corresponding attributes.
        """
        coords_cols = ["CDP_X", "CDP_Y"]
        bins_cols = ["INLINE_3D", "CROSSLINE_3D"]

        # Construct a mapping from bins to their coordinates and back
        bins_to_coords = self.get_headers(coords_cols + bins_cols)
        bins_to_coords = bins_to_coords.groupby(bins_cols, sort=False, as_index=False).agg("mean")
        bins_to_coords_reg = LinearRegression(copy_X=False, n_jobs=-1)
        bins_to_coords_reg.fit(bins_to_coords[bins_cols].to_numpy(), bins_to_coords[coords_cols].to_numpy())
        coords_to_bins_reg = LinearRegression(copy_X=False, n_jobs=-1)
        coords_to_bins_reg.fit(bins_to_coords[coords_cols].to_numpy(), bins_to_coords[bins_cols].to_numpy())

        # Compute geographic field contour
        geographic_contours = tuple(bins_to_coords_reg.predict(contour[:, 0])[:, None].astype(np.float32)
                                    for contour in self.bin_contours)
        perimeter = sum(cv2.arcLength(contour, closed=True) for contour in geographic_contours)

        # Set all geometry-related attributes
        self.has_inferred_geometry = True
        self._bins_to_coords_reg = bins_to_coords_reg
        self._coords_to_bins_reg = coords_to_bins_reg
        self.bin_size = np.diag(sp.linalg.polar(bins_to_coords_reg.coef_)[1])
        self.inline_length = (np.ptp(bins_to_coords["INLINE_3D"]) + 1) * self.bin_size[0]
        self.crossline_length = (np.ptp(bins_to_coords["CROSSLINE_3D"]) + 1) * self.bin_size[1]
        self.area = self.n_bins * np.prod(self.bin_size)
        self.perimeter = perimeter
        self.geographic_contours = geographic_contours
        self.is_2d = np.isclose(self.area, 0)

    @staticmethod
    def _cast_coords(coords, transformer):
        """Linearly convert `coords` from one coordinate system to another according to a passed `transformer`."""
        if transformer is None:
            raise ValueError("Survey geometry was not inferred, call `infer_geometry` method first.")
        coords = np.array(coords)
        is_coords_1d = coords.ndim == 1
        coords = np.atleast_2d(coords)
        transformed_coords = transformer.predict(coords)
        if is_coords_1d:
            return transformed_coords[0]
        return transformed_coords

    def coords_to_bins(self, coords):
        """Convert `coords` from geographic coordinate system to floating-valued bins.

        Notes
        -----
        Before calling this method, survey geometry must be inferred using :func:`~Survey.infer_geometry`.

        Parameters
        ----------
        coords : array-like with 2 elements or 2d array-like with shape (n_coords, 2)
            Geographic coordinates to be converted to bins.

        Returns
        -------
        bins : np.ndarray with 2 elements or 2d np.ndarray with shape (n_coords, 2)
            Floating-valued bin for each coordinate from `coords`. Has the same shape as `coords`.

        Raises
        ------
        ValueError
            If survey geometry was not inferred.
        """
        return self._cast_coords(coords, self._coords_to_bins_reg)

    def bins_to_coords(self, bins):
        """Convert `bins` to coordinates in geographic coordinate system.

        Notes
        -----
        Before calling this method, survey geometry must be inferred using :func:`~Survey.infer_geometry`.

        Parameters
        ----------
        bins : array-like with 2 elements or 2d array-like with shape (n_bins, 2)
            Bins to be converted to geographic coordinates.

        Returns
        -------
        coords : np.ndarray with 2 elements or 2d np.ndarray with shape (n_bins, 2)
            Floating-valued geographic coordinates for each bin from `bins`. Has the same shape as `bins`.

        Raises
        ------
        ValueError
            If survey geometry was not inferred.
        """
        return self._cast_coords(bins, self._bins_to_coords_reg)

    @staticmethod
    def _dist_to_contours(coords, contours):
        """Calculate minimum signed distance from points in `coords` to each contour in `contours`."""
        coords = np.array(coords, dtype=np.float32)
        is_coords_1d = coords.ndim == 1
        coords = np.atleast_2d(coords)
        dist = np.empty(len(coords), dtype=np.float32)
        for i, coord in enumerate(coords):
            dists = [cv2.pointPolygonTest(contour, coord, measureDist=True) for contour in contours]
            dist[i] = dists[np.abs(dists).argmin()]
        if is_coords_1d:
            return dist[0]
        return dist

    def dist_to_geographic_contours(self, coords):
        """Calculate signed distances from each of `coords` to the field contour in geographic coordinate system.

        Returned values may by positive (inside the contour), negative (outside the contour) or zero (on an edge).

        Notes
        -----
        Before calling this method, survey geometry must be inferred using :func:`~Survey.infer_geometry`.

        Parameters
        ----------
        coords : array-like with 2 elements or 2d array-like with shape (n_coords, 2)
            Geographic coordinates to estimate distance to field contour for.

        Returns
        -------
        dist : np.float32 or np.ndarray with shape (n_coords,)
            Signed distances from each of `coords` to the field contour in geographic coordinate system. Matches the
            length of `coords`.

        Raises
        ------
        ValueError
            If survey geometry was not inferred.
        """
        if self.geographic_contours is None:
            raise ValueError("Survey geometry was not inferred, call `infer_geometry` method first.")
        return self._dist_to_contours(coords, self.geographic_contours)

    def dist_to_bin_contours(self, bins):
        """Calculate signed distances from each of `bins` to the field contour in bin coordinate system.

        Returned values may by positive (inside the contour), negative (outside the contour) or zero (on an edge).

        Notes
        -----
        Before calling this method, properties of survey binning must be inferred using :func:`~Survey.infer_binning`.

        Parameters
        ----------
        bins : array-like with 2 elements or 2d array-like with shape (n_bins, 2)
            Bin coordinates to estimate distance to field contour for.

        Returns
        -------
        dist : np.float32 or np.ndarray with shape (n_bins,)
            Signed distances from each of `bins` to the field contour in bin coordinate system. Matches the length of
            `coords`.

        Raises
        ------
        ValueError
            If properties of survey binning were not inferred.
        """
        if self.bin_contours is None:
            raise ValueError("Properties of survey binning were not inferred, call `infer_binning` method first.")
        return self._dist_to_contours(bins, self.bin_contours)

    #------------------------------------------------------------------------#
    #                     Statistics computation methods                     #
    #------------------------------------------------------------------------#

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
        traces_pos = np.sort(get_cols(headers, "TRACE_SEQUENCE_FILE") - 1)
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
        traces_pos = get_cols(headers, "TRACE_SEQUENCE_FILE") - 1
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

        Each line of the file stores the first break time for a trace in the last column. The combination of all but
        the last columns should act as a unique trace identifier and is used to match the trace from the file with the
        corresponding trace in `self.headers`.

        The file can have any format that can be read by `pd.read_csv`, by default, it's expected to have
        whitespace-separated values.

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
            A survey with loaded times of first breaks.

        Raises
        ------
        ValueError
            If there is not a single match of rows from the file with those in `self.headers`.
        """
        self = maybe_copy(self, inplace, ignore="headers")  # pylint: disable=self-cls-assignment

        # If decimal is not provided, try inferring it from the first line
        if decimal is None:
            with open(path, 'r', encoding=encoding) as f:
                row = f.readline()
            decimal = '.' if '.' in row else ','

        trace_id_cols = to_list(trace_id_cols)
        file_columns = trace_id_cols + [first_breaks_col]
        first_breaks_df = pd.read_csv(path, delimiter=delimiter, names=file_columns, index_col=trace_id_cols,
                                      decimal=decimal, encoding=encoding, **kwargs)
        self.headers = self.headers.join(first_breaks_df, on=trace_id_cols, how="inner", rsuffix="_loaded")
        if self.is_empty:
            warnings.warn("Empty headers after first breaks loading", RuntimeWarning)
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

    def filter_by_metric(self, metric_name, threshold=None, inplace=False):
        """"Filter traces using metric with name `metric_name` and passed `threshold`.

        Parameters
        ----------
        metric_name : str
            Name of metric that stores in `self.qc_metrics`.
        threshold : int, optional, defaults to None
            Threshold to use during filtration. If None, theshold defined in metric will be used.
        inplace : bool, optional, defaults to False
            Whether to remove traces inplace or return a new survey instance.

        Returns
        -------
        Survey
            Filtered survey.
        """

        if not self.qc_metrics:
            raise ValueError("Not a single metric has been calculated yet, call `self.qc_tracewise` to compute one")

        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        metric = self.qc_metrics.get(metric_name)
        if metric is None:
            avalible_metrics = ', '.join(self.qc_metrics.keys())
            raise ValueError(f"`metric_name` must be one of {avalible_metrics}, but {metric_name} was given")
        self.filter(lambda metric_value: ~metric.binarize(metric_value, threshold) , cols=metric_name, inplace=True)

    def remove_dead_traces(self, header_name=None, chunk_size=1000, inplace=False, bar=True):
        """ Remove dead (constant) traces from the survey.
        Calculates :class:`~survey.metrics.DeadTrace` if `header_name` is not passed.

        Parameters
        ----------
        header_name : str, optional, defaults to None
            Name of header column with marked dead traces.
        chunk_size : int, optional, defaults to 1000
            Number of traces loaded on each iteration.
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
        if header_name is None:
            header_name = DeadTrace.name
            if header_name not in self.headers:
                self.qc_tracewise(DeadTrace, chunk_size=chunk_size, bar=bar)

        self.filter_by_metric(header_name, inplace=True)
        return self

    #------------------------------------------------------------------------#
    #                         Task specific methods                          #
    #------------------------------------------------------------------------#

    @staticmethod
    def _get_optimal_origin(arr, step):
        """Find a position in an array `arr` that maximizes sum of each `step`-th element from it to the end of the
        array. In case of multiple such positions, return the one closer to `step // 2`."""
        mod = len(arr) % step
        if mod:
            arr = np.pad(arr, (0, step - mod))
        step_sums = arr.reshape(-1, step).sum(axis=0)
        max_indices = np.nonzero(step_sums == step_sums.max())[0]
        return max_indices[np.abs(max_indices - step // 2).argmin()]

    def generate_supergathers(self, centers=None, origin=None, size=3, step=20, border_indent=0, strict=True,
                              reindex=True, inplace=False):
        """Combine several adjacent CDP gathers into ensembles called supergathers.

        Supergather generation is usually performed as a first step of velocity analysis. A substantially larger number
        of traces processed at once leads to increased signal-to-noise ratio: seismic wave reflections are much more
        clearly visible than on single CDP gathers and the velocity spectra calculated using
        :func:`~Gather.calculate_vertical_velocity_spectrum` are more coherent
        which allows for more accurate stacking velocity picking.

        The method creates two new `headers` columns called `SUPERGATHER_INLINE_3D` and `SUPERGATHER_CROSSLINE_3D`
        equal to `INLINE_3D` and `CROSSLINE_3D` of the central CDP gather. Note, that some gathers may be assigned to
        several supergathers at once and their traces will become duplicated in `headers`.

        Parameters
        ----------
        centers : 2d array-like with shape (n_supergathers, 2), optional
            Centers of supergathers being generated. If not given, calculated by the `origin` of a supergather grid.
            Measured in lines.
        origin : int or tuple of 2 ints, optional
            Origin of the supergather grid, used only if `centers` are not given. If `None`, generated automatically to
            maximize the number of supergathers. Measured in lines.
        size : int or tuple of 2 ints, optional, defaults to 3
            Supergather size along inline and crossline axes. Single int defines sizes for both axes. Measured in
            lines.
        step : int or tuple of 2 ints, optional, defaults to 20
            Supergather step along inline and crossline axes. Single int defines steps for both axes. Used to define a
            grid of supergathers if `centers` are not given. Measured in lines.
        border_indent : int, optional, defaults to 0
            Avoid placing supergather centers closer than this distance to the field contour. Used only if `centers`
            are not given. Measured in lines.
        strict : bool, optional, defaults to True
            If `True`, guarantees that each gather in a generated supergather will have at least one trace or, in other
            words, that the supergather entirely lies within the field. Used only if `centers` are not given.
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
        ValueError
            If supergathers have already been generated.
        """
        line_cols = ["INLINE_3D", "CROSSLINE_3D"]
        super_line_cols = ["SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"]
        if not self.has_inferred_binning:
            raise KeyError("INLINE_3D and CROSSLINE_3D headers must be loaded")
        if set(super_line_cols) <= self.available_headers:
            raise ValueError("Supergathers have already been generated")

        self = maybe_copy(self, inplace, ignore="headers")  # pylint: disable=self-cls-assignment
        size = np.broadcast_to(size, 2)
        step = np.broadcast_to(step, 2)

        if centers is None:
            # Erode the field mask according to border_indent and strict flags
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, np.broadcast_to(border_indent, 2) * 2 + 1).T
            field_mask = cv2.erode(self.field_mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if strict:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size).T
                field_mask = cv2.erode(field_mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            step = np.minimum(step, field_mask.shape)

            # Calculate origins of the supergather grid along inline and crossline directions
            if origin is not None:
                origin_i, origin_x = (np.broadcast_to(origin, 2) - self.field_mask_origin) % step
            else:
                origin_i = self._get_optimal_origin(field_mask.sum(axis=1), step[0])
                origin_x = self._get_optimal_origin(field_mask.sum(axis=0), step[1])

            # Calculate supergather centers by their grid
            grid_i = np.arange(origin_i, field_mask.shape[0], step[0])
            grid_x = np.arange(origin_x, field_mask.shape[1], step[1])
            centers = np.stack(np.meshgrid(grid_i, grid_x), -1).reshape(-1, 2)
            is_valid = field_mask[centers[:, 0], centers[:, 1]].astype(bool)
            centers = centers[is_valid] + self.field_mask_origin

        centers = np.array(centers)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("Passed centers must have shape (n_supergathers, 2)")

        # Construct a bridge table with mapping from supergather centers to their bins
        shifts_grid = np.meshgrid(np.arange(size[0]) - size[0] // 2, np.arange(size[1]) - size[1] // 2)
        shifts = np.stack(shifts_grid, axis=-1).reshape(-1, 2)
        bridge = np.column_stack([centers.repeat(size.prod(), axis=0), (centers[:, None] + shifts).reshape(-1, 2)])
        bridge = pd.DataFrame(bridge, columns=super_line_cols+line_cols)
        bridge.set_index(line_cols, inplace=True)

        headers = self.headers.join(bridge, on=line_cols, how="inner")
        if reindex:
            headers.reset_index(inplace=True)
            headers.set_index(super_line_cols, inplace=True)
            headers.sort_index(kind="stable", inplace=True)
        self.headers = headers
        return self

    def qc_tracewise(self, metrics=None, chunk_size=1000, n_workers=None, bar=True, overwrite=False):
        """Calculate tracewise QC metrics.

        Parameters
        ----------
        metrics : :class:`~metrics.TracewiseMetric`, or list of :class:`~metrics.TracewiseMetric` objects, optional
            Metric objects or instances that define metrics to calculate. If None, all metrics that can be initialized
            with reasonable default parameters are calculated.
        chunk_size : int, optional, defaults to 1000
            Number of traces loaded on each iteration.
        n_workers : int, optional
            The number of threads to be spawned to calculate metrics. Defaults to the number of cpu cores.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        Survey
            Survey with metrics written to headers and filled `self.qc_metrics` dict.

        Raises
        ------
        TypeError
            If provided metrics are not  :class:`~metrics.TracewiseMetric` subclasses
        """

        if metrics is None:
            metrics = DEFAULT_TRACEWISE_METRICS
        metrics, _ = initialize_metrics(metrics, metric_class=TracewiseMetric)

        metric_names = {metric.name for metric in metrics}
        if metric_names <= set(self.qc_metrics.keys()):
            msg = ', '.join(metric_names & set(self.qc_metrics.keys()))
            if not overwrite:
                raise ValueError(f"{msg} already calculated. Use `overwrite=True` or rename it.")
            warnings.warn(f'{msg} already calculated and will be rewritten.')

        n_chunks = self.n_traces // chunk_size + (1 if self.n_traces % chunk_size else 0)
        if n_workers is None:
            n_workers = os.cpu_count()
        n_workers = min(n_chunks, n_workers)

        idx_sort = self['TRACE_SEQUENCE_FILE'].argsort(kind='stable')
        orig_idx = idx_sort.argsort(kind='stable')

        # TODO: try to preallocate all memory before compliting the metrics calculation
        def calc_metrics(i, chunk_size):
            headers = self.headers.iloc[idx_sort[i * chunk_size: (i + 1) * chunk_size]]
            gather = self.load_gather(headers)
            results = {}
            for metric in metrics:
                header_cols = metric.header_cols
                if isinstance(metric, BaseWindowMetric):
                    metric = partial(metric, return_rms=False)
                results.update(zip(to_list(header_cols), np.atleast_2d(metric(gather))))
            return pd.DataFrame(results)

        # Precompile all numba decorated metrics to avoid hanging of the ThreadPoolExecutor during first metrics call
        _ = calc_metrics(0, 1)

        futures = []
        with tqdm(total=self.n_traces, desc="Traces processed", disable=not bar) as pbar:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for i in range(n_chunks):
                    future = pool.submit(calc_metrics, i, chunk_size)
                    future.add_done_callback(lambda fut: pbar.update(len(fut.result())))
                    futures.append(future)

        results = pd.concat([future.result() for future in futures], ignore_index=True, copy=False).iloc[orig_idx]
        results.index = self.headers.index
        self.headers[results.columns] = results
        self.qc_metrics.update({metric.name: metric for metric in metrics})
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

        Plotting must be performed in a JupyterLab environment with the `%matplotlib widget` magic executed and
        `ipympl` and `ipywidgets` libraries installed.

        Parameters
        ----------
        show_contour : bool, optional, defaults to True
            Whether to display a field contour if survey geometry was inferred.
        keep_aspect : bool, optional, defaults to False
            Whether to keep aspect ratio of the map plot.
        source_id_cols : str or list of str, optional
            Trace headers that uniquely identify a seismic source. If not given, `self.source_id_cols` is used.
        source_sort_by : str or list of str, optional
            Header names to sort the displayed common source gathers by. If not given, passed `sort_by` value is used.
        receiver_id_cols : str or list of str, optional
            Trace headers that uniquely identify a receiver. If not given, `self.receiver_id_cols` is used.
        receiver_sort_by : str or list of str, optional
            Header names to sort the displayed common receiver gathers by. If not given, passed `sort_by` value is
            used.
        sort_by : str or list of str, optional
            Default header names to sort the displayed gather by. If not given, no sorting is performed.
        gather_plot_kwargs : dict, optional
            Additional arguments to pass to `Gather.plot`.
        x_ticker : str or dict, optional
            Parameters to control `x` axis tick formatting and layout of the map plot. See `.utils.set_ticks` for more
            details.
        y_ticker : dict, optional
            Parameters to control `y` axis tick formatting and layout of the map plot. See `.utils.set_ticks` for more
            details.
        figsize : tuple with 2 elements, optional, defaults to (4.5, 4.5)
            Size of created map and gather figures. Measured in inches.
        orientation : {"horizontal", "vertical"}, optional, defaults to "horizontal"
            Defines whether to stack the main and auxiliary plots horizontally or vertically.
        kwargs : misc, optional
            Additional keyword arguments to pass to `matplotlib.axes.Axes.scatter` when plotting the map.
        """
        SurveyGeometryPlot(self, **kwargs).plot()

    def _construct_map(self, values, name, by, id_cols=None, drop_duplicates=False, agg=None, bin_size=None):
        """Construct a metric map of `values` aggregated by gather, whose type is defined by `by`."""
        index_cols, coords_cols = get_cols_from_by(self, by)
        index_cols = get_first_defined(id_cols, index_cols)

        metric_data = self.get_headers(coords_cols)
        if index_cols is not None:
            index_cols = to_list(index_cols)
            metric_data[index_cols] = self[index_cols]
        metric_data[name] = values
        if drop_duplicates:
            metric_data.drop_duplicates(inplace=True)
        index = metric_data[index_cols] if index_cols is not None else None
        coords = metric_data[coords_cols]
        values = metric_data[name]

        metric = SurveyAttribute(name=name).provide_context(survey=self)
        return metric.construct_map(coords, values, index=index, agg=agg, bin_size=bin_size)

    def construct_header_map(self, col, by, id_cols=None, drop_duplicates=False, agg=None, bin_size=None):
        """Construct a metric map of trace header values aggregated by gather.

        Examples
        --------
        Construct a map of maximum offset by shots:
        >>> max_offset_map = survey.construct_header_map("offset", by="shot", agg="max")
        >>> max_offset_map.plot()

        The map allows for interactive plotting: a gather type defined by `by` will be displayed on click on the map.
        The gather may optionally be sorted if `sort_by` argument is passed to the `plot` method:
        >>> max_offset_map.plot(interactive=True, sort_by="offset")

        Parameters
        ----------
        col : str
            Headers column to extract values from.
        by : {"source", "shot", "receiver", "rec", "cdp", "cmp", "midpoint", "bin", "supergather"}
            Gather type to aggregate header values over.
        id_cols : str or list of str, optional
            Trace headers that uniquely identify a gather of the chosen type. Acts as an index of the resulting map.
        drop_duplicates : bool, optional, defaults to False
            Whether to drop duplicated entries of (index, coordinates, metric value). Useful when dealing with a header
            defined for a shot or receiver, not a trace (e.g. constructing a map of elevations by shots).
        agg : str or callable, optional, defaults to "mean"
            An aggregation function. Passed directly to `pandas.core.groupby.DataFrameGroupBy.agg`.
        bin_size : int, float or array-like with length 2, optional
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.

        Returns
        -------
        header_map : BaseMetricMap
            Constructed header map.
        """
        return self._construct_map(self[col], name=col, by=by, id_cols=id_cols, drop_duplicates=drop_duplicates,
                                   agg=agg, bin_size=bin_size)

    def construct_fold_map(self, by, id_cols=None, agg=None, bin_size=None):
        """Construct a metric map which stores the number of traces for each gather (fold).

        Examples
        --------
        Generate supergathers and calculate their fold:
        >>> supergather_columns = ["SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"]
        >>> supergather_survey = survey.generate_supergathers(size=7, step=7)
        >>> fold_map = supergather_survey.construct_fold_map(by="supergather")
        >>> fold_map.plot()

        Parameters
        ----------
        by : {"source", "shot", "receiver", "rec", "cdp", "cmp", "midpoint", "bin", "supergather"}
            Gather type to aggregate header values over.
        id_cols : str or list of str, optional
            Trace headers that uniquely identify a gather of the chosen type. Acts as an index of the resulting map.
        agg : str or callable, optional, defaults to "mean"
            An aggregation function. Passed directly to `pandas.core.groupby.DataFrameGroupBy.agg`.
        bin_size : int, float or array-like with length 2, optional
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.

        Returns
        -------
        fold_map : BaseMetricMap
            Constructed fold map.
        """
        tmp_map = self._construct_map(np.ones(self.n_traces), name="fold", by=by, id_cols=id_cols, agg="sum")
        index = tmp_map.index_data[tmp_map.index_cols]
        coords = tmp_map.index_data[tmp_map.coords_cols]
        values = tmp_map.index_data[tmp_map.metric_name]
        return tmp_map.metric.construct_map(coords, values, index=index, agg=agg, bin_size=bin_size)

    def construct_qc_maps(self, by, metric_names=None, id_cols=None, agg=None, bin_size=None):
        """Construct a map of tracewise metric aggregated by gathers.

        Parameters
        ----------
        by : {"source", "shot", "receiver", "rec", "cdp", "cmp", "midpoint", "bin", "supergather"}
            Gather type to aggregate header values over.
        metric : str or list of str, optional
            name(s) of metrics to build metrics maps. If None, maps for all metrics that were calculated for this
            survey are built.
        agg : str or callable, optional, defaults to "mean"
            An aggregation function. Passed directly to `pandas.core.groupby.DataFrameGroupBy.agg`.
        bin_size : int, float or array-like with length 2, optional
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.

        Returns
        -------
        BaseMetricMap
            Constructed metric map.
        """
        squeeze_output = isinstance(metric_names, str)
        if metric_names is None:
            metric_names = list(self.qc_metrics.keys())
        metric_names = to_list(metric_names)

        metrics = []
        for metric_name in metric_names:
            if "/" in metric_name:
                metric_list = [self.qc_metrics[name.strip()] for name in metric_name.split("/")]
                metric = MetricsRatio(*metric_list)
            elif metric_name not in self.qc_metrics:
                raise ValueError(f'Metric with name "{metric_name}" is not calculated yet!')
            else:
                metric = self.qc_metrics[metric_name]
            metrics.append(metric.provide_context(survey=self))

        index_cols, coords_cols = get_cols_from_by(self, by)
        index_cols = get_first_defined(id_cols, index_cols)
        coords = self.get_headers(coords_cols)
        index = self.get_headers(index_cols) if index_cols is not None else coords

        mmaps = []
        for metric in metrics:
            metric_mmap = metric.construct_map(coords, self.get_headers(metric.header_cols), index=index, agg=agg,
                                               bin_size=bin_size)
            mmaps.append(metric_mmap)
        return mmaps[0] if squeeze_output else mmaps
