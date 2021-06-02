""" File with gather class. """
import os
import warnings
from copy import deepcopy
from textwrap import dedent

import segyio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .muting import Muter
from .semblance import Semblance, ResidualSemblance
from .velocity_cube import StackingVelocity, VelocityCube
from .decorators import batch_method
from .utils import to_list, convert_times_to_mask, convert_mask_to_pick


class Gather:
    """ !! """
    def __init__(self, headers, data, samples, sample_rate, survey):
        self.headers = headers
        self.data = data
        self.samples = samples
        self.sample_rate = sample_rate
        self.survey = survey
        self.sort_by = None
        self.mask = None

    @property
    def times(self):
        return self.samples

    @property
    def offsets(self):
        return self.headers['offset'].values

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, key):
        return self.headers[key].values

    def __setitem__(self, key, value):
        key = to_list(key)
        val = pd.DataFrame(value, columns=key, index=self.headers.index)
        self.headers[key] = val

    def __str__(self):
        # Calculating offset range
        offsets = self.headers.get('offset')
        min_offset, max_offset = np.min(offsets), np.max(offsets)

        # Determining index value
        index = np.unique(self.headers.index)
        index = 'combined' if len(index) > 1 else index.item()

        # Counting the number of zero/constant traces
        n_dead_traces = np.isclose(np.max(self.data, axis=1), np.min(self.data, axis=1)).sum()

        msg = f"""
        Parent survey path:          {self.survey.path}
        Parent survey name:          {self.survey.name}

        Number of traces:            {self.data.shape[0]}
        Trace length:                {len(self.samples)} samples
        Sample rate:                 {self.sample_rate} ms
        Times range:                 [{min(self.samples)} ms, {max(self.samples)} ms]
        Offsets range:               [{min_offset} m, {max_offset} m]

        Index name(s):               {', '.join(self.headers.index.names)}
        Index value:                 {index}
        Gather sorting:              {self.sort_by}

        Gather statistics:
        Number of dead traces:       {n_dead_traces}
        mean | std:                  {np.mean(self.data):>10.2f} | {np.std(self.data):<10.2f}
         min | max:                  {np.min(self.data):>10.2f} | {np.max(self.data):<10.2f}
         q01 | q99:                  {self.get_quantile(0.01):>10.2f} | {self.get_quantile(0.99):<10.2f}
        """
        return dedent(msg)

    def info(self):
        print(self)

    def get_coords(self, coords_columns="index"):
        if coords_columns is None:
            return (None, None)
        if coords_columns == "index":
            coords_columns = self.headers.index.names
        coords = np.unique(self.headers.reset_index()[coords_columns].values, axis=0)
        if coords.shape[0] != 1:
            raise ValueError("Gather coordinates are non-unique")
        if coords.shape[1] != 2:
            raise ValueError(f"Gather position must be defined by exactly two coordinates, not {coords.shape[1]}")
        return tuple(coords[0].tolist())

    @batch_method(target='for', copy=False)
    def copy(self):
        survey = self.survey
        self.survey = None
        self_copy = deepcopy(self)
        self_copy.survey = survey
        self.survey = survey
        return self_copy

    def _validate_header_cols(self, required_header_cols):
        headers = self.headers.reset_index()
        required_header_cols = to_list(required_header_cols)
        if any(col not in headers for col in required_header_cols):
            err_msg = "The following headers must be preloaded: {}"
            raise ValueError(err_msg.format(", ".join(required_header_cols)))

    def _validate_sorting(self, required_sorting):
        if self.sort_by != required_sorting:
            raise ValueError(f"Gather should be sorted by {required_sorting} not {self.sort_by}")

    def validate(self, required_header_cols=None, required_sorting=None):
        if required_header_cols is not None:
            self._validate_header_cols(required_header_cols)
        if required_sorting is not None:
            self._validate_sorting(required_sorting)
        return self

    #------------------------------------------------------------------------#
    #                              Dump methods                              #
    #------------------------------------------------------------------------#

    @batch_method(target='for', force=True, copy=False)
    def dump(self, path, name=None, copy_header=False):
        parent_handler = self.survey.segy_handler

        if name is None:
            name = "_".join(map(str, [self.survey.name] + to_list(self.headers.index.values[0])))
        if not os.path.splitext(name)[1]:
            name += '.sgy'
        full_path = os.path.join(path, name)

        os.makedirs(path, exist_ok=True)
        # Create segyio spec. We choose only specs that relate to unstructured data.
        spec = segyio.spec()
        spec.samples = self.samples
        spec.ext_headers = parent_handler.ext_headers
        spec.format = parent_handler.format
        spec.tracecount = len(self.data)

        trace_headers = self.headers.reset_index()

        # Remember ordinal numbers of traces in parent segy to further copy their headers
        # and reset them to start from 1 in the resulting file to match segy standard.
        trace_ids = trace_headers[self.survey.TRACE_ID_HEADER].values - 1
        trace_headers[self.survey.TRACE_ID_HEADER] = np.arange(len(trace_headers)) + 1

        # Keep only headers, relevant to segy file.
        used_header_names = set(trace_headers.columns) & set(segyio.tracefield.keys.keys())
        trace_headers = trace_headers[used_header_names]

        # Now we change column name's into byte number based on the segy standard.
        trace_headers.rename(columns=lambda col_name: segyio.tracefield.keys[col_name], inplace=True)
        trace_headers_dict = trace_headers.to_dict('index')

        with segyio.create(full_path, spec) as dump_handler:
            # Copy binary headers from parent segy. This is possibly incorrect and needs to be checked
            # if the number of traces or sample ratio changes.
            # TODO: Check if bin headers matter
            dump_handler.bin = parent_handler.bin

            # Copy textual headers from parent segy.
            for i in range(spec.ext_headers + 1):
                dump_handler.text[i] = parent_handler.text[i]

            # Dump traces and their headers. Optionally copy headers from parent segy.
            dump_handler.trace = self.data
            for i, dump_h in trace_headers_dict.items():
                if copy_header:
                    dump_handler.header[i].update(parent_handler.header[trace_ids[i]])
                dump_handler.header[i].update(dump_h)
        return self

    #------------------------------------------------------------------------#
    #                         Normalization methods                          #
    #------------------------------------------------------------------------#

    def _apply_agg_func(self, func, tracewise, **kwargs):
        axis = 1 if tracewise else None
        return func(self.data, axis=axis, **kwargs)

    def get_quantile(self, q, tracewise=False, use_global=False):
        if use_global:
            return self.survey.get_quantile(q)
        quantiles = self._apply_agg_func(func=np.quantile, tracewise=tracewise, q=q)
        # return the same type as q in case of global calculation: either single float or array-like
        return quantiles.item() if not tracewise and quantiles.ndim == 0 else quantiles

    @batch_method(target='for')
    def scale_standard(self, tracewise=True, use_global=False, eps=1e-10):
        if use_global:
            if not self.survey.has_stats:
                raise ValueError('Global statistics were not calculated, call `Survey.collect_stats` first.')
            mean = self.survey.mean
            std = self.survey.std
        else:
            mean = self._apply_agg_func(func=np.mean, tracewise=tracewise, keepdims=True)
            std = self._apply_agg_func(func=np.std, tracewise=tracewise, keepdims=True)

        self.data = (self.data - mean) / (std + eps)
        return self

    @batch_method(target='for')
    def scale_maxabs(self, q_min=0, q_max=1, tracewise=False, use_global=False, clip=False, eps=1e-10):
        min_value, max_value = self.get_quantile([q_min, q_max], tracewise=tracewise, use_global=use_global)
        max_abs = np.maximum(np.abs(min_value), np.abs(max_value))
        # Use np.atleast_2d(array).T to make the array 2-dimentional by adding dummy trailing axes
        # for further broadcasting to work tracewise
        self.data /= np.atleast_2d(max_abs).T + eps
        if clip:
            self.data = np.clip(self.data, 0, 1)
        return self

    @batch_method(target='for')
    def scale_minmax(self, q_min=0, q_max=1, tracewise=False, use_global=False, clip=False, eps=1e-10):
        min_value, max_value = self.get_quantile([q_min, q_max], tracewise=tracewise, use_global=use_global)
        # Use np.atleast_2d(array).T to make the array 2-dimentional by adding dummy trailing axes
        # for further broadcasting to work tracewise
        min_value = np.atleast_2d(min_value).T
        max_value = np.atleast_2d(max_value).T
        self.data = (self.data - min_value) / (max_value - min_value + eps)
        if clip:
            self.data = np.clip(self.data, 0, 1)
        return self

    #------------------------------------------------------------------------#
    #                    First-breaks processing methods                     #
    #------------------------------------------------------------------------#

    @batch_method(target="for")
    def pick_to_mask(self, first_breaks_col='FirstBreak'):
        self.validate(required_header_cols=first_breaks_col)
        self.mask = convert_times_to_mask(times=self[first_breaks_col], sample_rate=self.sample_rate,
                                          mask_length=self.shape[1])
        return self

    @batch_method(target='for')
    def mask_to_pick(self, threshold=0.5, first_breaks_col='FirstBreak'):
        if self.mask is None:
            raise ValueError('Save mask to self.mask component.')
        self[first_breaks_col] = convert_mask_to_pick(self.mask, self.sample_rate, threshold)
        return self

    #------------------------------------------------------------------------#
    #                         Gather muting methods                          #
    #------------------------------------------------------------------------#

    @batch_method(target="for", copy=False)
    def create_muter(self, mode="first_breaks", **kwargs):
        if mode == "first_breaks":
            first_breaks_col = kwargs.pop("first_breaks_col", "FirstBreak")
            return Muter(mode=mode, offsets=self.offsets, times=self[first_breaks_col], **kwargs)
        return Muter(mode=mode, **kwargs)

    @batch_method(target="threads", args_to_unpack="muter")
    def mute(self, muter):
        self.data *= convert_times_to_mask(times=muter(self.offsets), sample_rate=self.sample_rate,
                                           mask_length=self.shape[1])
        return self

    #------------------------------------------------------------------------#
    #                     Semblance calculation methods                      #
    #------------------------------------------------------------------------#

    @batch_method(target="threads", copy=False)
    def calculate_semblance(self, velocities, win_size=25):
        self.validate(required_sorting="offset")
        return Semblance(gather=self, velocities=velocities, win_size=win_size)

    @batch_method(target="for", args_to_unpack="stacking_velocity", copy=False)
    def calculate_residual_semblance(self, stacking_velocity, n_velocities=140, win_size=25, relative_margin=0.2):
        self.validate(required_sorting="offset")
        return ResidualSemblance(gather=self, stacking_velocity=stacking_velocity, n_velocities=n_velocities,
                                 win_size=win_size, relative_margin=relative_margin)

    #------------------------------------------------------------------------#
    #                       Gather processing methods                        #
    #------------------------------------------------------------------------#

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
    def get_central_cdp(self):
        self.validate(required_header_cols=["INLINE_3D", "SUPERGATHER_INLINE_3D",
                                            "CROSSLINE_3D", "SUPERGATHER_CROSSLINE_3D"])
        headers = self.headers.reset_index()
        mask = ((headers["SUPERGATHER_INLINE_3D"] == headers["INLINE_3D"]) &
                (headers["SUPERGATHER_CROSSLINE_3D"] == headers["CROSSLINE_3D"])).values
        self.headers = self.headers.loc[mask]
        self.data = self.data[mask]
        return self

    @batch_method(target="for")
    def apply_nmo(self, stacking_velocity, coords_columns="index"):
        if isinstance(stacking_velocity, VelocityCube):
            stacking_velocity = stacking_velocity.get_stacking_velocity(*self.get_coords(coords_columns))
        if not isinstance(stacking_velocity, StackingVelocity):
            raise ValueError("Only VelocityCube or StackingVelocity instances can be passed as a stacking_velocity")
        velocities_ms = stacking_velocity(self.times) / 1000  # from m/s to m/ms
        res = []
        for time, velocity in zip(self.times, velocities_ms):
            res.append(Semblance.apply_nmo(self.data.T, time, self.offsets, velocity, self.sample_rate))
        self.data = np.stack(res).T.astype(np.float32)
        return self

    @batch_method(target="for")
    def stack_gather(self):
        line_cols = ["INLINE_3D", "CROSSLINE_3D"]
        self.validate(required_header_cols=line_cols)
        headers = self.headers.reset_index()[line_cols].drop_duplicates()
        if len(headers) != 1:
            raise ValueError("Only a single CDP gather can be stacked")
        self.headers = headers.set_index(line_cols)
        self.headers[self.survey.TRACE_ID_HEADER] = 1

        # TODO: avoid zeros in semblance calculation
        self.data[self.data == 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.data = np.nanmean(self.data, axis=0, keepdims=True)
        self.data = np.nan_to_num(self.data)
        return self

    #------------------------------------------------------------------------#
    #                         Visualization methods                          #
    #------------------------------------------------------------------------#

    @batch_method(target="for")
    def plot(self, figsize=(10, 7), **kwargs):
        default_kwargs = {
            'cmap': 'gray',
            'vmin': np.quantile(self.data, 0.1),
            'vmax': np.quantile(self.data, 0.9),
            'aspect': 'auto',
        }
        default_kwargs.update(kwargs)
        plt.figure(figsize=figsize)
        plt.imshow(self.data.T, **default_kwargs)
        plt.show()
        return self
