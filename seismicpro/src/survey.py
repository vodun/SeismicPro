import os

import segyio
import numpy as np
import pandas as pd

from .gather import Gather
from .abstract_classes import AbstractSurvey


DEFAULT_HEADERS = ['offset', ]
TRACE_UID_COLUMN = 'TraceSequenceNumber'


class Survey(AbstractSurvey):
    def __init__(self, path, header_index=None, header_cols=None, limits=None, **kwargs):
        self.path = path
        self.name = os.path.basename(self.path).split('.')[0]
        # self.name = name if name is not None else self.basename
        self.headers = None
        self.limits = slice(limits)

        self.header_index = header_index
        self.index_len = len(self.header_index)
        #TODO: add default_headers to self.header_cols
        self.header_cols = header_cols

        self.segy_handler = segyio.open(self.path, ignore_geometry=True)
        self.segy_handler.mmap()

        # Get attributes from segy.
        self.sample_rate = segyio.dt(self.segy_handler) / 1000
        self.samples = self.segy_handler.samples[self.limits]

        headers = {}
        for column in self.header_cols:
            headers[column] = self.segy_handler.attributes(getattr(segyio.TraceField, column))[self.limits]

        headers = pd.DataFrame(headers)
        headers.reset_index(inplace=True)
        headers.rename(columns={'index': TRACE_UID_COLUMN}, inplace=True)

        headers.set_index(self.header_index, inplace=True)
        # To optimize futher sampling from mulitiindex.
        self.headers = headers.sort_index()

    def __del__(self):
        self.segy_handler.close()

    def get_gather(self, index=None):
        if index is None:
            index = self.headers.index[0]
        gather_headers = self.headers.xs(index, axis=0, drop_level=False).reset_index()
        load_indices = gather_headers[TRACE_UID_COLUMN].values

        data = np.stack(self.load_trace(idx) for idx in load_indices)

        gather = Gather(data=data, header_cols=gather_headers)
        return gather

    def load_trace(self, index):
        res = self.segy_handler.trace.raw[int(index) - 1][self.limits]
        return res

    def dump(self):
        pass

    def merge(self):
        pass

    def concat(self):
        pass

    def find_sdc_params(self):
        pass

    def find_equalization_params(self):
        pass
