import os

import segyio
import numpy as np
import pandas as pd

from .gather import Gather
from .utils import to_list


DEFAULT_HEADERS = {'offset',}


class Survey:
    """ !! """
    TRACE_ID_HEADER = 'TRACE_SEQUENCE_FILE'
    def __init__(self, path, header_index, header_cols=None, name=None):
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
        load_headers = set(header_index) | header_cols | DEFAULT_HEADERS

        # We always reconstruct this column, so there is no need to load it.
        if self.TRACE_ID_HEADER in load_headers:
            load_headers.remove(self.TRACE_ID_HEADER)

        self.segy_handler = segyio.open(self.path, ignore_geometry=True)
        self.segy_handler.mmap()

        # Get attributes from segy.
        self.sample_rate = segyio.dt(self.segy_handler) / 1000
        self.samples = self.segy_handler.samples
        self.samples_length = len(self.samples)

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
        self.is_trace_index = (header_index == to_list(self.TRACE_ID_HEADER))

    def __del__(self):
        self.segy_handler.close()

    def get_gather(self, index=None, limits=None, copy_headers=True):
        if not isinstance(limits, slice):
            limits = slice(*to_list(limits))
        limits = limits.indices(self.samples_length)
        trace_length = len(range(*limits))
        if trace_length == 0:
            raise ValueError('Trace length must be positive.')

        # Avoid time-consuming reset_index in case of iteration over individual traces
        # Use slicing instead of indexing to guarantee that DataFrame is always returned
        if self.is_trace_index:
            gather_headers = self.headers.iloc[index - 1 : index - 1]
            trace_indices = [index - 1]
        else:
            gather_headers = self.headers.loc[index:index]
            trace_indices = gather_headers.reset_index()[self.TRACE_ID_HEADER].values - 1

        if copy_headers:
            gather_headers = gather_headers.copy()
        data = np.stack([self.load_trace(i, limits, trace_length) for i in trace_indices])
        gather = Gather(headers=gather_headers, data=data, survey=self)
        return gather

    def sample_gather(self, limits=None, copy_headers=True):
        # TODO: write normal sampler here
        index = np.random.choice(self.headers.index)
        gather = self.get_gather(index=index, limits=limits, copy_headers=copy_headers)
        return gather

    def load_trace(self, index, limits, trace_length):
        """limits is an array [from, to]"""
        buf = np.empty(trace_length, dtype=np.float32)
        # Args for segy loader are following:
        #   * Buffer to write trace ampltudes
        #   * Index of loading trace
        #   * Unknown arg (always 1)
        #   * Unknown arg (always 1)
        #   * Position from which to start loading the trace
        #   * Position where to end loading
        #   * Step
        #   * Number of overall samples.
        res = self.segy_handler.xfd.gettr(buf, index, 1, 1, *limits, trace_length)
        return res

    def reindex(self, new_index):
        self.headers.reset_index(inplace=True)
        self.headers.set_index(new_index, inplace=True)
        # TODO: add sort and update self.is_trace_index
        return self

    def find_sdc_params(self):
        pass

    def find_equalization_params(self):
        pass
