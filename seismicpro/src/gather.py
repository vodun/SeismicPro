""" File with gather class. """
import os

import segyio
import numpy as np
import matplotlib.pyplot as plt

from .abstract_classes import AbstractGather

TRACE_ID_HEADER = 'TRACE_SEQUENCE_FILE'
FILE_EXT = '.sgy'

class Gather(AbstractGather):
    """ !! """
    def __init__(self, headers, data, survey=None):
        self.headers = headers
        self.survey = survey
        self.data = data
        self.sort_by = None

    def __getattr__(self):
        pass

    def dump(self, path, name=None):
        # TODO: Check does file.bin header matter?
        parent_handler = self.survey.segy_handler

        if name is None:
            gather_name = str(self.headers.index.value[0])
            name = survey.name + '_' + gather_name + FILE_EXT

        name = name + FILE_EXT if len(os.path.splitext(name)[1]) == 0 else name

        full_path = os.path.join(path, name)

        # Create segyio spec.
        spec = segyio.spec()
        spec.samples = parent_handler.samples
        spec.ext_headers = parent_handler.ext_headers
        spec.format = parent_handler.format
        spec.tracecount = len(self.data)

        # TODO: Can any operations change the order of the rows in the dataframe?
        unindex_headers = self.headers.reset_index()
        all_header_names = set(segyio.tracefield.keys.keys())
        used_header_names = set(unindex_headers.columns) & all_header_names

        # We are interested only in trace headers, thus we took only them.
        trace_headers = unindex_headers[used_header_names]
        # We need to fill this column because the trace order minght be changed during processing.
        trace_headers[TRACE_ID_HEADER]= np.arange(len(trace_headers)) + 1
        # Now we change column name's into byte number based on the segy standard.
        trace_headers.rename(columns=lambda col_name: segyio.tracefield.keys[col_name], inplace=True)
        trace_headers_dict = trace_headers.to_dict('index')

        with segyio.create(full_path, spec) as dump_handler:
            # Copy binary headers from parent segy.
            for i in range(spec.ext_headers + 1):
                dump_handler.text[i] = parent_handler.text[i]

            #!!This approach is possibly incorrect and needs to be checked when number of traces or samples ratio
            # changes.
            dump_handler.bin = parent_handler.bin

            # Save traces and trace headers
            dump_handler.trace = self.data
            dump_handler.header = parent_handler.header
            # Update trace headers from self.headers.
            for i, dump_h in trace_headers_dict.items():
                dump_handler.header[i].update(dump_h)


    def sort(self, by):
        if not isinstance(by, str):
            raise TypeError('`by` should be str, not {}'.format(type(by)))

        arg = np.argsort(self.headers[by].values, kind='stable')

        self.sort_by = by
        self.data = self.data[arg]
        self.headers = self.headers.iloc[arg]
        return self

    def equalize(self):
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

    def plot(self):
        kwargs = {
            'cmap': 'gray',
            'vmin': np.quantile(self.data, 0.1),
            'vmax': np.quantile(self.data, 0.9),
            'aspect': 'auto',
        }
        plt.figure(figsize=(10, 7))
        plt.imshow(self.data.T, **kwargs)


    def plot_gain(self):
        pass

    def plot_spectrum(self):
        pass

    def plot_stats(self):
        pass
