import os

import segyio
import numpy as np
import pandas as pd

from .gather import Gather
from .abstract_classes import AbstractSurvey

DEFAULT_HEADERS = ['offset', ]
TRACE_UID_HEADERS = ['TRACE_SEQUENCE_FILE']


class Survey(AbstractSurvey):
    def __init__(self, path, index_headers=None, headers=None, name=None, limits=None, **kwargs):
        self.path = path
        self.basename = os.path.basename(self.path).split('.')[0]
        self.name = name if name is not None else self.basename
        self.dataframe = None
        self.limits = slice(limits)

        self.index_headers = index_headers
        self.index_len = len(self.index_headers)
        #TODO: add default_headers to self.headers
        self.headers = headers
        self.headers = set(self.headers) | set(TRACE_UID_HEADERS)

        self.segy_handler = segyio.open(self.path, ignore_geometry=True)
        self.segy_handler.mmap()
        self.sample_rate = segyio.dt(self.segy_handler) / 1000

        dataframe = {}
        for column in self.headers:
            dataframe[column] = self.segy_handler.attributes(getattr(segyio.TraceField, column))[self.limits]

        dataframe = pd.DataFrame(dataframe)
        self.dataframe = dataframe.set_index(self.index_headers)
        # To optimize futher sampling from mulitiindex
        self.dataframe = self.dataframe.sort_index()

    def __del__(self):
        self.segy_handler.close()

    def get_gather(self, name=None, index=None):
        if index is None:
            index = self.dataframe.index[0]
        gather_dataframe = self.dataframe.xs(index, axis=0, drop_level=False).reset_index()
        load_indices = gather_dataframe[TRACE_UID_HEADERS[0]].values

        data = np.stack(self.load_trace(idx) for idx in load_indices)
        gather = Gather(data=data, headers=gather_dataframe)
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
