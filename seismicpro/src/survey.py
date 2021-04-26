import os
from copy import copy, deepcopy

import segyio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.interpolate import interp1d

from .gather import Gather
from .utils import to_list, calculate_stats, create_supergather_index
from .decorators import add_inplace_arg


class Survey:
    """ !! """
    TRACE_ID_HEADER = 'TRACE_SEQUENCE_FILE'
    DEFAULT_HEADERS = {'offset', }

    def __init__(self, path, header_index, header_cols=None, name=None, stats_limits=None, collect_stats=False, **kwargs):
        self.path = path
        basename = os.path.splitext(os.path.basename(self.path))[0]
        self.name = name if name is not None else basename

        if header_cols is None:
            header_cols = set()
        elif header_cols == 'all':
            header_cols = set(segyio.tracefield.keys.keys())
        else:
            header_cols = set(to_list(header_cols))

        header_index = to_list(header_index)
        load_headers = set(header_index) | header_cols | self.DEFAULT_HEADERS

        # We always reconstruct this column, so there is no need to load it.
        if self.TRACE_ID_HEADER in load_headers:
            load_headers.remove(self.TRACE_ID_HEADER)

        self.segy_handler = segyio.open(self.path, ignore_geometry=True)
        self.segy_handler.mmap()

        # Get attributes from segy.
        self.sample_rate = segyio.dt(self.segy_handler) / 1000
        self.file_samples = self.segy_handler.samples

        # Define samples and samples length that belongs to particular `stats_limits`.
        self.stats_limits = None
        self.samples = None
        self.samples_length = None
        self.set_limits(stats_limits)

        headers = {}
        for column in load_headers:
            headers[column] = self.segy_handler.attributes(segyio.tracefield.keys[column])[:]

        headers = pd.DataFrame(headers).reset_index()
        # TODO: add why do we use unknown column
        headers.rename(columns={'index': self.TRACE_ID_HEADER}, inplace=True)
        headers[self.TRACE_ID_HEADER] += 1
        headers.set_index(header_index, inplace=True)
        # To optimize futher sampling from mulitiindex.
        self.headers = headers.sort_index()

        # Precalculate survey statistics
        self.has_stats = False
        self.min = None
        self.max = None
        self.mean = None
        self.std = None
        self.quantile_interpolator = None
        if collect_stats:
            self.collect_stats(**kwargs)

    def __del__(self):
        self.segy_handler.close()

    def __getstate__(self):
        state = copy(self.__dict__)
        state["segy_handler"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.segy_handler = segyio.open(self.path, ignore_geometry=True)
        self.segy_handler.mmap()

    def copy(self):
        return deepcopy(self)

    @add_inplace_arg
    def filter(self, header_cols, cond, axis=None, *args, **kwargs):
        headers = self.headers[to_list(header_cols)]
        if axis is None:
            mask = cond(headers, *args, **kwargs)
        else:
            mask = headers.apply(cond, axis=axis, raw=True, args=args, **kwargs)
        self.headers = self.headers.loc[mask.values]
        return self

    def set_limits(self, limits):
        self.stats_limits = self._process_limits(limits)
        self.samples = self.file_samples[slice(*self.stats_limits)]
        self.samples_length = len(self.samples)
        if self.samples_length == 0:
            raise ValueError('Trace length must be positive.')

    def _process_limits(self, limits):
        if not isinstance(limits, slice):
            limits = slice(*to_list(limits))
        return limits.indices(len(self.file_samples))

    def get_gather(self, index=None, limits=None, copy_headers=True):
        limits = self.stats_limits if limits is None else self._process_limits(limits)
        samples = self.file_samples[slice(*limits)]
        trace_length = len(samples)

        gather_headers = self.headers.loc[index]
        if copy_headers:
            gather_headers = gather_headers.copy()
        trace_indices = gather_headers.reset_index()[self.TRACE_ID_HEADER].values - 1

        data = np.empty((len(trace_indices), trace_length), dtype=np.float32)
        for i, ix in enumerate(trace_indices):
            self.load_trace(buf=data[i], index=ix, limits=limits, trace_length=trace_length)
        # TODO: slice samples in gather by limits
        gather = Gather(headers=gather_headers, data=data, samples=samples, survey=self)
        return gather

    def sample_gather(self, limits=None, copy_headers=True):
        # TODO: write normal sampler here
        index = np.random.choice(self.headers.index)
        gather = self.get_gather(index=index, limits=limits, copy_headers=copy_headers)
        return gather

    def load_trace(self, buf, index, limits, trace_length):
        # Args for segy loader are following:
        #   * Buffer to write trace ampltudes
        #   * Index of loading trace
        #   * Unknown arg (always 1)
        #   * Unknown arg (always 1)
        #   * Position from which to start loading the trace
        #   * Position where to end loading
        #   * Step
        #   * Number of overall samples.
        return self.segy_handler.xfd.gettr(buf, index, 1, 1, *limits, trace_length)

    def load_picking(self, path, picking_col='Picking'):
        segy_columns = ['FieldRecord', 'TraceNumber']
        picking_columns = segy_columns + [picking_col]
        picking_df = pd.read_csv(path, names=picking_columns, delim_whitespace=True, decimal=',')

        headers = self.headers.reset_index()
        missing_cols = set(segy_columns) - set(headers)
        if missing_cols:
            raise ValueError(f'Missing {missing_cols} column(s) required for picking loading.')

        headers = headers.merge(picking_df, on=segy_columns)
        if headers.empty:
            raise ValueError('Empty headers after picking loading.')
        headers.set_index(self.headers.index.names, inplace=True)
        self.headers = headers.sort_index()
        return self

    @add_inplace_arg
    def reindex(self, new_index):
        self.headers.reset_index(inplace=True)
        self.headers.set_index(new_index, inplace=True)
        self.headers.sort_index(inplace=True)
        return self

    @add_inplace_arg
    def generate_supergathers(self, size=(3, 3), step=(20, 20), modulo=(0, 0), reindex=True):
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

    def collect_stats(self, dataset=None, n_samples=100000, quantile_precision=2, bar=True):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        headers = self.headers if dataset is None else dataset.index.headers
        traces_pos = headers.reset_index()['TRACE_SEQUENCE_FILE'].values
        np.random.shuffle(traces_pos)

        global_min, global_max = np.inf, -np.inf
        global_sum, global_sq_sum = 0, 0
        traces_length = 0
        traces_buf = np.empty((n_samples, self.samples_length), dtype=np.float32)
        trace = np.empty(self.samples_length, dtype=np.float32)

        # Accumulate min, max, mean and std values of survey traces
        for i, pos in tqdm(enumerate(traces_pos), desc="Calculating statistics",
                           total=len(traces_pos), disable=not bar):
            self.load_trace(index=pos-1, limits=self.stats_limits,
                            trace_length=self.samples_length, buf=trace)
            trace_min, trace_max, trace_sum, trace_sq_sum = calculate_stats(trace)
            global_min = min(trace_min, global_min)
            global_max = max(trace_max, global_max)
            global_sum += trace_sum
            global_sq_sum += trace_sq_sum
            traces_length += len(trace)

            # Sample random traces to calculate approximate quantiles
            if i < n_samples:
                traces_buf[i] = trace

        self.min = global_min
        self.max = global_max
        self.mean = global_sum / traces_length
        self.std = np.sqrt((global_sq_sum / traces_length) - (global_sum / traces_length)**2)

        # Calculate all q-quantiles from 0 to 1 with step 1 / 10**quantile_precision
        q = np.round(np.linspace(0, 1, num=10**quantile_precision), decimals=quantile_precision)
        quantiles = np.quantile(traces_buf.ravel(), q=q)
        # 0 and 1 quantiles are replaced with actual min and max values respectively
        quantiles[0], quantiles[-1] = global_min, global_max
        self.quantile_interpolator = interp1d(q, quantiles)

        self.has_stats = True
        return self

    def get_quantile(self, q):
        if not self.has_stats:
            raise ValueError('Global statistics were not calculated, call `Survey.collect_stats` first.')
        quantiles = self.quantile_interpolator(q)
        # return the same type as q: either single float or array-like
        return quantiles.item() if quantiles.ndim == 0 else quantiles
