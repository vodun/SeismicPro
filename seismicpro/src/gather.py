""" File with gather class. """
import os
import copy
import warnings

import segyio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
        self.mask = None

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

    @property
    def shape(self):
        return self.data.shape

    @batch_method
    def copy(self):
        survey = self.survey
        self.survey = None
        self_copy = copy.deepcopy(self)
        self_copy.survey = survey
        self.survey = survey
        return self_copy

    @batch_method(force=True)
    def dump(self, path, name=None, copy_header=False):
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
                if copy_header:
                    dump_handler.header[i].update(parent_handler.header[trace_ids[i]])
                dump_handler.header[i].update(dump_h)
        return self

    @batch_method(target="threads")
    def sort(self, by):
        if not isinstance(by, str):
            raise TypeError('`by` should be str, not {}'.format(type(by)))
        order = np.argsort(self.headers[by].values, kind='stable')
        self.sort_by = by
        self.data = self.data[order]
        self.headers = self.headers.iloc[order]
        return self

    @batch_method(target="for")
    def create_picking_mask(self):
        if 'Picking' not in self.headers.columns:
            raise ValueError('Load picking first.')
        picking_ixs = np.around(self['Picking'] / self.sample_rate).astype(np.int32)
        mask = (np.arange(self.shape[1]).reshape(1, -1) - picking_ixs.reshape(-1, 1)) > 0
        self.mask = np.int32(mask.squeeze())
        return self

    @batch_method(target="threads")
    def mute(self, muting):
        self.data = self.data * muting.create_mask(trace_len=self.shape[1], offsets=self.offsets,
                                                   sample_rate=self.sample_rate)
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

    @batch_method(target="for")
    def get_central_cdp(self):
        headers = self.headers.reset_index()
        line_cols = ["INLINE_3D", "SUPERGATHER_INLINE_3D", "CROSSLINE_3D", "SUPERGATHER_CROSSLINE_3D"]
        if any(col not in headers for col in line_cols):
            raise ValueError("The method can be applied only for supergathers")
        mask = ((headers["SUPERGATHER_INLINE_3D"] == headers["INLINE_3D"]) &
                (headers["SUPERGATHER_CROSSLINE_3D"] == headers["CROSSLINE_3D"])).values
        self.headers = self.headers.loc[mask]
        self.data = self.data[mask]
        return self

    @batch_method(target="for")
    def correct_gather(self, stacking_velocities):
        velocity_interpolator = interp1d(*zip(*stacking_velocities), fill_value="extrapolate")
        velocities = velocity_interpolator(self.samples) / 1000
        res = []
        for time, velocity in zip(self.samples, velocities):
            res.append(Semblance.base_calc_nmo(self.data.T, time, self.offsets, velocity, self.sample_rate))
        self.data = np.stack(res).T.astype(np.float32)
        return self

    @batch_method(target="for")
    def stack_gather(self):
        headers = self.headers.reset_index()
        line_cols = ["INLINE_3D", "CROSSLINE_3D"]
        if any(col not in headers for col in line_cols):
            raise ValueError("The method can be applied only for CDP gathers")
        headers = headers[line_cols].drop_duplicates()
        if len(headers) != 1:
            raise ValueError("Only a single CDP gather can be stacked")
        self.headers = headers.set_index(line_cols)
        self.headers[self.survey.TRACE_ID_HEADER] = 0

        # TODO: avoid zeros in semblance calculation
        self.data[self.data == 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.data = np.nanmean(self.data, axis=0, keepdims=True)
        self.data = np.nan_to_num(self.data)
        return self

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
