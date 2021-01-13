import os

import segyio
import numpy as np
import pandas as pd

from .gather import Gather
from .abstract_classes import AbstractSurvey

from ..batchflow.utils import is_iterable

DEFAULT_HEADERS = ['offset', ]
TRACE_UID_COLUMN = 'TraceSequenceNumber'


class Survey(AbstractSurvey):
    """ !! """
    def __init__(self, path, header_index=None, header_cols=None, limits=None, name=None, **kwargs):
        self.path = path
        self.headers = None
        self.limits = slice(limits)
        basename = os.path.basename(self.path).split('.')[0]
        self.name = name if name is not None else basename

        if header_cols == 'all':
            header_cols = [header_name.__str__() for header_name in segyio.TraceField.enums()]

        header_index = (header_index, ) if not is_iterable(header_index) else header_index
        header_cols = (header_cols, ) if not is_iterable(header_cols) else header_cols
        load_headers = set(header_index) | set(header_cols)

        self.segy_handler = segyio.open(self.path, ignore_geometry=True)
        self.segy_handler.mmap()

        # Get attributes from segy.
        self.sample_rate = segyio.dt(self.segy_handler) / 1000
        self.samples = self.segy_handler.samples[self.limits]

        headers = {}
        for column in load_headers:
            headers[column] = self.segy_handler.attributes(getattr(segyio.TraceField, column))[self.limits]

        headers = pd.DataFrame(headers)
        headers.reset_index(inplace=True)
        headers.rename(columns={'index': TRACE_UID_COLUMN}, inplace=True)
        headers.set_index(list(header_index), inplace=True)
        # To optimize futher sampling from mulitiindex.
        self.headers = headers.sort_index()

    def __del__(self):
        self.segy_handler.close()

    def get_gather(self, index=None, random=False):
        if index is None:
            # TODO: Write normal random choice.
            index = self.headers.index[0] if random else np.random.choice(self.headers.index)

        # Here we use
        gather_headers = self.headers.loc[[index]].reset_index()
        data = np.stack([self.load_trace(idx) for idx in gather_headers[TRACE_UID_COLUMN]])

        gather = Gather(data=data, headers=gather_headers)
        return gather

    def load_trace(self, index):
        res = self.segy_handler.trace.raw[int(index)][self.limits]
        return res

    def dump(self):
        pass

    def merge(self): # delete
        pass

    def concat(self):
        pass

    def find_sdc_params(self):
        pass

    def find_equalization_params(self):
        pass
