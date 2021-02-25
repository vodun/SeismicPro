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

    def __del__(self):
        self.segy_handler.close()

    def get_gather(self, index=None, limits=None):
        if not isinstance(limits, slice):
            limits = slice(*to_list(limits))
        limits = limits.indices(self.samples_length)
        trace_length = len(range(*limits))
        if trace_length == 0:
            raise ValueError('Trace length must be positive.')

        # TODO: description why do we use [index] instead of index.
        gather_headers = self.headers.loc[[index]].copy()
        data = np.stack([self.load_trace(idx-1, limits, trace_length) for idx in gather_headers[self.TRACE_ID_HEADER]])
        gather = Gather(headers=gather_headers, data=data, survey=self)
        return gather

    def sample_gather(self, limits=None):
        # TODO: write normal sampler here
        index = np.random.choice(self.headers.index)
        gather = self.get_gather(index=index, limits=limits)
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
        return self

    def find_sdc_params(self):
        pass

    def find_equalization_params(self):
        pass
