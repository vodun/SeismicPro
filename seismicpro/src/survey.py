import os

import segyio
import numpy as np
import pandas as pd

from .gather import Gather
from .abstract_classes import AbstractSurvey

from ..batchflow.utils import is_iterable


DEFAULT_HEADERS = ['offset', ]
TRACE_ID_HEADER = 'TRACE_SEQUENCE_FILE'


class Survey(AbstractSurvey):
    """ !! """
    def __init__(self, path, header_index, header_cols=None, name=None, **kwargs):
        self.path = path
        self.headers = None
        basename = os.path.splitext(os.path.basename(self.path))[0]
        self.name = name if name is not None else basename

        if header_cols == 'all':
            header_cols = tuple(segyio.tracefield.keys.keys())

        header_index = (header_index, ) if not is_iterable(header_index) else header_index
        header_cols = (header_cols, ) if not is_iterable(header_cols) else header_cols
        load_headers = set(header_index) | set(header_cols)
        # We always reconstruct this column, so there is no need to load it.
        if TRACE_ID_HEADER in load_headers:
            load_headers.remove(TRACE_ID_HEADER)

        self.segy_handler = segyio.open(self.path, ignore_geometry=True)
        self.segy_handler.mmap()

        # Get attributes from segy.
        self.sample_rate = segyio.dt(self.segy_handler) / 1000
        self.samples = self.segy_handler.samples

        headers = {}
        for column in load_headers:
            headers[column] = self.segy_handler.attributes(segyio.tracefield.keys[column])[:]

        headers = pd.DataFrame(headers)
        headers.reset_index(inplace=True)
        # TODO: add why do we use unknown column
        headers.rename(columns={'index': TRACE_ID_HEADER}, inplace=True)
        headers.set_index(list(header_index), inplace=True)
        # To optimize futher sampling from mulitiindex.
        self.headers = headers.sort_index()

    def __del__(self):
        self.segy_handler.close()

    def get_gather(self, index=None, limits=None):
        if index is None:
            index = self.headers.index[0]
        # TODO: description why do we use [index] instead of index.
        gather_headers = self.headers.loc[[index]].copy()
        data = np.stack([self.load_trace(idx, limits) for idx in gather_headers[TRACE_ID_HEADER]])
        gather = Gather(data=data, headers=gather_headers, survey=self)
        return gather

    def sample_gather(self, limits=None):
        # TODO: write normal sampler here
        index = np.random.choice(self.headers.index)
        gather = get_gather(self, index=index, limits=limits)
        return gather

    def load_trace(self, index, limits=None):
        """limits is an array [from, to]"""
        limits = limits if limits is not None else [0, len(self.samples)]
        trace_length = limits[1] - limits[0]

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
        res = self.segy_handler.xfd.gettr(buf, index, 1, 1, *limits, 1, trace_length)
        return res

    def dump(self):
        """ params to dump:
            1. spec
            2. ext_headers (aka. self.segy_handler.text)
            3. header
            4. bin headers (aka. bin)
        """
        pass

    def merge(self): # delete
        pass

    def concat(self):
        pass

    def find_sdc_params(self):
        pass

    def find_equalization_params(self):
        pass
