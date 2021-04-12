""" File with gather class. """
import os
import copy
import warnings

import segyio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import to_list, convert_mask_to_pick
from .decorators import batch_method, validate_gather
from .semblance import Semblance, ResidualSemblance
from .velocity_cube import VelocityLaw, VelocityCube


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

    def _get_unique_header_val(self, header):
        headers = self.headers.reset_index()
        if header not in headers:
            return None
        unique_vals = np.unique(headers[header])
        if len(unique_vals) > 1:
            return None
        return unique_vals.item()

    @property
    def inline(self):
        return self._get_unique_header_val("INLINE_3D")

    @property
    def crossline(self):
        return self._get_unique_header_val("CROSSLINE_3D")

    @property
    def supergather_inline(self):
        return self._get_unique_header_val("SUPERGATHER_INLINE_3D")

    @property
    def supergather_crossline(self):
        return self._get_unique_header_val("SUPERGATHER_CROSSLINE_3D")

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
    def pick_to_mask(self, picking_col='Picking'):
        if picking_col not in self.headers:
            raise ValueError('Picking is not loaded.')
        picking_ixs = np.around(self[picking_col] / self.sample_rate).astype(np.int32) - 1
        mask = (np.arange(self.shape[1]) - picking_ixs.reshape(-1, 1)) > 0
        self.mask = mask.astype(np.int32)
        return self

    @batch_method(target='for')
    def mask_to_pick(self, threshold=0.5, picking_col='Picking'):
        if self.mask is None:
            raise ValueError('Save mask to self.mask component.')
        self[picking_col] = convert_mask_to_pick(self.mask, threshold) * self.sample_rate
        return self

    @batch_method(target="threads")
    def mute(self, muting):
        self.data = self.data * muting.create_mask(trace_len=self.shape[1], offsets=self.offsets,
                                                   sample_rate=self.sample_rate)
        return self

    @batch_method(target="threads")
    @validate_gather(required_sorting="offset")
    def calculate_semblance(self, velocities, win_size=25):
        return Semblance(gather=self.data, times=self.samples, offsets=self.offsets,
                         velocities=velocities, win_size=win_size,
                         inline=self.supergather_inline, crossline=self.supergather_crossline)

    @batch_method(target='for')
    @validate_gather(required_sorting="offset")
    def calculate_residual_semblance(self, stacking_velocities, num_vels=140, win_size=25, relative_margin=0.2):
        return ResidualSemblance(gather=self.data, times=self.samples, offsets=self.offsets,
                                 stacking_velocities=stacking_velocities, num_vels=num_vels, win_size=win_size,
                                 relative_margin=relative_margin)

    @batch_method(target="for")
    @validate_gather(required_header_cols=["INLINE_3D", "SUPERGATHER_INLINE_3D",
                                           "CROSSLINE_3D", "SUPERGATHER_CROSSLINE_3D"])
    def get_central_cdp(self):
        headers = self.headers.reset_index()
        mask = ((headers["SUPERGATHER_INLINE_3D"] == headers["INLINE_3D"]) &
                (headers["SUPERGATHER_CROSSLINE_3D"] == headers["CROSSLINE_3D"])).values
        self.headers = self.headers.loc[mask]
        self.data = self.data[mask]
        return self

    @batch_method(target="for")
    def correct_gather(self, velocity_model):
        if isinstance(velocity_model, VelocityCube):
            velocity_model = velocity_model.get_law(self.inline, self.crossline)
        if not isinstance(velocity_model, VelocityLaw):
            raise ValueError("Only VelocityCube or VelocityLaw instances can be passed as a velocity_model")
        velocities = velocity_model(self.samples) / 1000
        res = []
        for time, velocity in zip(self.samples, velocities):
            res.append(Semblance.base_calc_nmo(self.data.T, time, self.offsets, velocity, self.sample_rate))
        self.data = np.stack(res).T.astype(np.float32)
        return self

    @batch_method(target="for")
    @validate_gather(required_header_cols=["INLINE_3D", "CROSSLINE_3D"])
    def stack_gather(self):
        line_cols = ["INLINE_3D", "CROSSLINE_3D"]
        headers = self.headers.reset_index()[line_cols].drop_duplicates()
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

    def _apply_agg_func(self, func, tracewise, **kwargs):
        axis = 1 if tracewise else None
        return func(self.data, axis=axis, keepdims=True, **kwargs)

    @batch_method(target='for')
    def scale_standard(self, tracewise=True, use_global=False, eps=1e-10):
        if use_global:
            if self.survey.mean is None or self.survey.std is None:
                err_msg = "Global statistics were not calculated, set `use_global` to `False` " \
                          "or call `Survey.collect_stats` first."
                raise ValueError(err_msg)
            mean = self.survey.mean
            std = self.survey.std
        else:
            mean = self._apply_agg_func(func=np.mean, tracewise=tracewise)
            std = self._apply_agg_func(func=np.std, tracewise=tracewise)

        self.data = (self.data - mean) / (std + eps)
        return self

    @batch_method(target='for')
    def scale_maxabs(self, q_min=0, q_max=1, tracewise=True, use_global=False, clip=False, eps=1e-10):
        if use_global:
            min_value = self.survey.get_quantile(q_min)
            max_value = self.survey.get_quantile(q_max)
        else:
            min_value = self._apply_agg_func(func=np.quantile, tracewise=tracewise, q=q_min)
            max_value = self._apply_agg_func(func=np.quantile, tracewise=tracewise, q=q_max)

        max_abs = np.maximum(np.abs(min_value), np.abs(max_value))
        self.data /= max_abs + eps
        if clip:
            self.data = np.clip(self.data, 0, 1)
        return self

    @batch_method(target='for')
    def scale_minmax(self, q_min=0, q_max=1, tracewise=True, use_global=False, clip=False, eps=1e-10):
        if use_global:
            min_value = self.survey.get_quantile(q_min)
            max_value = self.survey.get_quantile(q_max)
        else:
            min_value = self._apply_agg_func(func=np.quantile, tracewise=tracewise, q=q_min)
            max_value = self._apply_agg_func(func=np.quantile, tracewise=tracewise, q=q_max)

        self.data = (self.data - min_value) / (max_value - min_value + eps)
        if clip:
            self.data = np.clip(self.data, 0, 1)
        return self

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
