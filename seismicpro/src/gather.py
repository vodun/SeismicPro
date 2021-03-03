""" File with gather class. """
import os
import copy

import segyio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import to_list
from .decorators import batch_method
from .semblance import Semblance, ResidualSemblance


class Gather:
    """ !! """
    def __init__(self, headers, data, survey):
        self.headers = headers
        self.data = data
        self.survey = survey
        self.samples = survey.samples
        self.sample_rate = survey.sample_rate
        self.sort_by = None

    def __getitem__(self, key):
        return self.headers[key].values

    def __setitem__(self, key, value):
        key = to_list(key)
        val = pd.DataFrame(value, columns=key, index=self.headers.index)
        self.headers[key] = val

    @property
    def offsets(self):
        return self.headers['offset'].values

    @property
    def index(self):
        return self.headers.index.values[0]

    @batch_method
    def copy(self):
        survey = self.survey
        self.survey = None
        self_copy = copy.deepcopy(self)
        self_copy.survey = survey
        self.survey = survey
        return self_copy

    @batch_method(force=True)
    def dump(self, path, name=None):
        # TODO: Check does file.bin header matters?
        parent_handler = self.survey.segy_handler

        if name is None:
            name = "_".join(map(str, [self.survey.name] + to_list(self.index)))
        if os.path.splitext(name)[1] == "":
            name += '.sgy'
        full_path = os.path.join(path, name)

        # Create segyio spec. We choose only specs that relate to unstructured data.
        spec = segyio.spec()
        spec.samples = self.samples
        spec.ext_headers = parent_handler.ext_headers
        spec.format = parent_handler.format
        spec.tracecount = len(self.data)

        trace_headers = self.headers.reset_index()
        # We need to save start index in segy file in order to save correct header.
        trace_ids = trace_headers[self.survey.TRACE_ID_HEADER].values
        # Select only the loaded headers from dataframe.
        used_header_names = set(trace_headers.columns) & set(segyio.tracefield.keys.keys())
        trace_headers = trace_headers[used_header_names]

        # We need to fill this column because the trace order minght be changed during the processing.
        trace_headers[self.survey.TRACE_ID_HEADER] = np.arange(len(trace_headers)) + 1

        # Now we change column name's into byte number based on the segy standard.
        trace_headers.rename(columns=lambda col_name: segyio.tracefield.keys[col_name], inplace=True)
        trace_headers_dict = trace_headers.to_dict('index')

        with segyio.create(full_path, spec) as dump_handler:
            # Copy binary headers from parent segy.
            for i in range(spec.ext_headers + 1):
                dump_handler.text[i] = parent_handler.text[i]

            # This is possibly incorrect and needs to be checked when number of traces or samples ratio changes.
            dump_handler.bin = parent_handler.bin

            # Save traces and trace headers.
            dump_handler.trace = self.data
            # Update trace headers from self.headers.
            for i, dump_h in trace_headers_dict.items():
                dump_handler.header[i].update(parent_handler.header[trace_ids[i]])
                dump_handler.header[i].update(dump_h)

    @batch_method(target="threads")
    def sort(self, by):
        if not isinstance(by, str):
            raise TypeError('`by` should be str, not {}'.format(type(by)))
        order = np.argsort(self.headers[by].values, kind='stable')
        self.sort_by = by
        self.data = self.data[order]
        self.headers = self.headers.iloc[order]
        return self

    @batch_method(target='for')
    def calculate_semblance(self, velocities, win_size=25):
        if self.sort_by != 'offset':
            raise ValueError(f'Gather should be sorted by `offset` not {self.sort_by}.')
        return Semblance(gather=self.data, times=self.samples, offsets=self.offsets,
                         velocities=velocities, win_size=win_size)

    @batch_method(target='for')
    def calculate_residual_semblance(self, stacking_velocities, num_vels=140, win_size=25, relative_margin=0.2):
        if self.sort_by != 'offset':
            raise ValueError(f'Gather should be sorted by `offset` not {self.sort_by}.')
        return ResidualSemblance(gather=self.data, times=self.samples, offsets=self.offsets,
                                 stacking_velocities=stacking_velocities, num_vels=num_vels, win_size=win_size,
                                 relative_margin=relative_margin)

    def equalize(self, attr):
        pass

    def band_pass_filter(self):
        pass

    def correct_spherical_divergence(self):
        pass

    def drop_zero_traces(self):
        pass

    def hodograph_straightening(self):
        pass

    def mcm(self):
        pass

    def pad_traces(self):
        pass

    def slice_traces(self):
        pass

    @batch_method(target="for")
    def plot(self):
        kwargs = {
            'cmap': 'gray',
            'vmin': np.quantile(self.data, 0.1),
            'vmax': np.quantile(self.data, 0.9),
            'aspect': 'auto',
        }
        plt.figure(figsize=(10, 7))
        plt.imshow(self.data.T, **kwargs)
        return self

    def plot_gain(self):
        pass

    def plot_spectrum(self):
        pass

    def plot_stats(self):
        pass
