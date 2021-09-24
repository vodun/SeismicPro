"""Implements Gather class that represents a group of seismic traces that share some common acquisition parameter"""

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
from .utils import to_list, convert_times_to_mask, convert_mask_to_pick, mute_gather, normalization, correction


class Gather:
    """A class representing a single seismic gather.

    A gather is a collection of seismic traces that share some common acquisition parameter (same index value of the
    generating survey header in our case). Unlike `Survey`, `Gather` instance stores loaded seismic traces along with
    a corresponding subset of its parent survey header.

    `Gather` instance can be created in three main ways:
    1. Either by calling `Survey.sample_gather` to get a randomly selected gather,
    2. Or by calling `Survey.get_gather` to get a particular gather by its index value,
    3. Or by calling `Index.get_gather` to get a particular gather by its index value from a specified survey.

    Most of the methods change gather data inplace, thus `Gather.copy` may come in handy to keep the original gather
    available.

    Examples
    --------
    Let's load a randomly selected common source gather, sort it by offset and plot:
    >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
    >>> gather = survey.sample_gather().sort(by="offset")
    >>> gather.plot()

    Parameters
    ----------
    headers : pd.DataFrame
        A subset of parent survey header with common index value defining the gather.
    data : 2d np.ndarray
        Trace data of the gather with (num_traces, trace_lenght) layout.
    samples : 1d np.ndarray of floats
        Recording time for each trace value. Measured in milliseconds.
    sample_rate : float
        Sample rate of seismic traces. Measured in milliseconds.
    survey : Survey
        A survey that generated the gather.

    Attributes
    ----------
    headers : pd.DataFrame
        A subset of parent survey header with common index value defining the gather.
    data : 2d np.ndarray
        Trace data of the gather with (num_traces, trace_lenght) layout.
    samples : 1d np.ndarray of floats
        Recording time for each trace value. Measured in milliseconds.
    sample_rate : float
        Sample rate of seismic traces. Measured in milliseconds.
    survey : Survey
        A survey that generated the gather.
    sort_by : None or str
        Headers column that was used for gather sorting. If `None`, no sorting was performed.
    """
    def __init__(self, headers, data, samples, sample_rate, survey):
        self.headers = headers
        self.data = data
        self.samples = samples
        self.sample_rate = sample_rate
        self.survey = survey
        self.sort_by = None

    @property
    def times(self):
        """1d np.ndarray of floats: Recording time for each trace value. Measured in milliseconds."""
        return self.samples

    @property
    def offsets(self):
        """1d np.ndarray of floats: The distance between source and receiver for each trace. Measured in meters."""
        return self.headers['offset'].values

    @property
    def shape(self):
        """tuple with 2 elements: The number of traces in the gather and trace length in samples."""
        return self.data.shape

    def __getitem__(self, key):
        """Select gather headers by their names.

        Parameters
        ----------
        key : str or list of str
            Gather headers to get.

        Returns
        -------
        headers : np.ndarray
            Headers values.
        """
        return self.headers[key].values

    def __setitem__(self, key, value):
        """Set given values to selected gather headers.

        Parameters
        ----------
        key : str or list of str
            Gather headers to set values for.
        value : np.ndarray
            Headers values to set.
        """
        key = to_list(key)
        val = pd.DataFrame(value, columns=key, index=self.headers.index)
        self.headers[key] = val

    def __str__(self):
        """Print gather metadata including information about its survey, headers and traces."""

        # Calculate offset range
        offsets = self.headers.get('offset')
        offset_range = f'[{np.min(offsets)} m, {np.max(offsets)} m]' if offsets is not None else None

        # Determine index value
        index = np.unique(self.headers.index)
        index = 'combined' if len(index) > 1 else index.item()

        # Count the number of zero/constant traces
        n_dead_traces = np.isclose(np.max(self.data, axis=1), np.min(self.data, axis=1)).sum()
        msg = f"""
        Parent survey path:          {self.survey.path}
        Parent survey name:          {self.survey.name}

        Number of traces:            {self.data.shape[0]}
        Trace length:                {len(self.samples)} samples
        Sample rate:                 {self.sample_rate} ms
        Times range:                 [{min(self.samples)} ms, {max(self.samples)} ms]
        Offsets range:               {offset_range}

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
        """Print gather metadata including information about its survey, headers and traces."""
        print(self)

    def get_coords(self, coords_columns="index"):
        """Get spatial coordinates of the gather.

        Parameters
        ----------
        coords_columns : None, "index" or 2 element array-like, defaults to "index"
            - If `None`, (`None`, `None`) tuple is returned.
            - If "index", unique index value is used to define gather coordinates
            - If 2 element array-like, `coords_columns` define gather headers to get x and y coordinates from.
            In the last two cases index or column values are supposed to be unique for all traces in the gather.

        Returns
        -------
        coords : tuple with 2 elements
            Gather spatial coordinates.

        Raises
        ------
        ValueError
            If gather coordinates are non-unique or more than 2 columns were passed.
        """
        if coords_columns is None:
            return (None, None)
        if coords_columns == "index":
            coords_columns = list(self.headers.index.names)
        coords = np.unique(self.headers.reset_index()[coords_columns].values, axis=0)
        if coords.shape[0] != 1:
            raise ValueError("Gather coordinates are non-unique")
        if coords.shape[1] != 2:
            raise ValueError(f"Gather position must be defined by exactly two coordinates, not {coords.shape[1]}")
        return tuple(coords[0].tolist())

    @batch_method(target='threads', copy_src=False)
    def copy(self):
        """Perform a deepcopy of all gather attributes except for `survey`, which is kept unchanged.

        Returns
        -------
        copy : Gather
            Copy of the gather.
        """
        survey = self.survey
        self.survey = None
        self_copy = deepcopy(self)
        self_copy.survey = survey
        self.survey = survey
        return self_copy

    def _validate_header_cols(self, required_header_cols):
        """Check if the gather headers contain all columns from `required_header_cols`."""
        headers = self.headers.reset_index()
        required_header_cols = to_list(required_header_cols)
        if any(col not in headers for col in required_header_cols):
            err_msg = "The following headers must be preloaded: {}"
            raise ValueError(err_msg.format(", ".join(required_header_cols)))

    def _validate_sorting(self, required_sorting):
        """Check if the gather is sorted by `required_sorting` header."""
        if self.sort_by != required_sorting:
            raise ValueError(f"Gather should be sorted by {required_sorting} not {self.sort_by}")

    def validate(self, required_header_cols=None, required_sorting=None):
        """Perform the following checks for a gather:
            1. Its header contains all columns from `required_header_cols`,
            2. It is sorted by `required_sorting` header.

        Parameters
        ----------
        required_header_cols : None or str or array-like of str, defaults to None
            Required gather header columns. If `None`, no check is performed.
        required_sorting : None or str, defaults to None
            Required gather sorting. If `None`, no check is performed.

        Returns
        -------
        self : Gather
            Self unchanged.

        Raises
        ------
        ValueError
            If any of checks above failed.
        """
        if required_header_cols is not None:
            self._validate_header_cols(required_header_cols)
        if required_sorting is not None:
            self._validate_sorting(required_sorting)
        return self

    #------------------------------------------------------------------------#
    #                              Dump methods                              #
    #------------------------------------------------------------------------#

    @batch_method(target='for', force=True, copy_src=False)
    def dump(self, path, name=None, copy_header=False):
        """Save the gather to a `.sgy` file.

        Notes
        -----
        All binary and textual headers are copied from the parent SEG-Y file unchanged.

        Parameters
        ----------
        path : str
            The directory to dump the gather in.
        name : str, optional, defaults to None
            The name of the file. If `None`, the concatenation of the survey name and the value of gather index will
            be used.
        copy_header : bool, optional, defaults to False
            Whether to copy the headers that weren't loaded during Survey creation from the parent SEG-Y file.

        Returns
        -------
        self : Gather
            Gather unchanged.

        Raises
        ------
        ValueError
            If empty `name` was specified.
        """
        parent_handler = self.survey.segy_handler

        if name is None:
            name = "_".join(map(str, [self.survey.name] + to_list(self.headers.index.values[0])))
        if name == "":
            raise ValueError("Argument `name` can not be empty.")
        if not os.path.splitext(name)[1]:
            name += ".sgy"
        full_path = os.path.join(path, name)

        os.makedirs(path, exist_ok=True)
        # Create segyio spec. We choose only specs that relate to unstructured data.
        spec = segyio.spec()
        spec.samples = self.samples
        spec.ext_headers = parent_handler.ext_headers
        spec.format = parent_handler.format
        spec.tracecount = len(self.data)

        trace_headers = self.headers.reset_index()

        # Remember ordinal numbers of traces in the parent SEG-Y file to further copy their headers
        # and reset them to start from 1 in the resulting file to match SEG-Y standard.
        trace_ids = trace_headers["TRACE_SEQUENCE_FILE"].values - 1
        trace_headers["TRACE_SEQUENCE_FILE"] = np.arange(len(trace_headers)) + 1

        # Keep only headers, defined by SEG-Y standard.
        used_header_names = set(trace_headers.columns) & set(segyio.tracefield.keys.keys())
        trace_headers = trace_headers[used_header_names]

        # Now we change column name's into byte number based on the SEG-Y standard.
        trace_headers.rename(columns=lambda col_name: segyio.tracefield.keys[col_name], inplace=True)
        trace_headers_dict = trace_headers.to_dict("index")

        with segyio.create(full_path, spec) as dump_handler:
            # Copy binary headers from the parent SEG-Y file. This is possibly incorrect and needs to be checked
            # if the number of traces or sample ratio changes.
            # TODO: Check if bin headers matter
            dump_handler.bin = parent_handler.bin

            # Copy textual headers from the parent SEG-Y file.
            for i in range(spec.ext_headers + 1):
                dump_handler.text[i] = parent_handler.text[i]

            # Dump traces and their headers. Optionally copy headers from the parent SEG-Y file.
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
        """Apply a `func` either to entire gather's data or to each trace independently.

        Notes
        -----
        `func` must accept an `axis` argument.

        Parameters
        ----------
        func : callable
            Function to be applied to the gather's data.
        tracewise : bool
            If `True`, the `func` is applied to each trace independently, otherwise to the entire gather's data.
        kwargs : misc, optional
            Additional keyword arguments to `func`.

        Returns
        -------
        result : misc
            The result of the application of the `func` to the gather's data.
        """
        axis = 1 if tracewise else None
        return func(self.data, axis=axis, **kwargs)

    def get_quantile(self, q, tracewise=False, use_global=False):
        """Calculate the `q`-th quantile of the gather or fetch the global quantile from the parent survey.

        Notes
        -----
        The `tracewise` mode is only available when `use_global` is `False`.

        Parameters
        ----------
        q : float or array-like of floats
            Quantile or a sequence of quantiles to compute, which must be between 0 and 1 inclusive.
        tracewise : bool, optional, default False
            If `True`, the quantiles are computed for each trace independently, otherwise for the entire gather.
        use_global : bool, optional, default False
            If `True`, the survey's quantiles are used, otherwise the quantiles are computed for the gather data.

        Returns
        -------
        q : float or array-like of floats
            The `q`-th quantile values.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        if use_global:
            return self.survey.get_quantile(q)
        quantiles = self._apply_agg_func(func=np.nanquantile, tracewise=tracewise, q=q).astype(np.float32)
        # return the same type as q in case of global calculation: either single float or array-like
        return quantiles.item() if not tracewise and quantiles.ndim == 0 else quantiles

    @batch_method(target='threads')
    def scale_standard(self, tracewise=True, use_global=False, eps=1e-10):
        r"""Standardize the gather by removing the mean and scaling to unit variance.

        The standard score of a gather `g` is calculated as:
        :math:`G = \frac{g - m}{s + eps}`,
        where:
        `m` - the mean of the gather or global average if `use_global=True`,
        `s` - the standard deviation of the gather or global standard deviation if `use_global=True`,
        `eps` - a constant that is added to the denominator to avoid division by zero.

        Notes
        -----
        1. The presence of NaN values in the gather will lead to incorrect behavior of the scaler.
        2. Standardization is performed inplace.

        Parameters
        ----------
        tracewise : bool, optional, defaults to True
            If `True`, mean and standard deviation are calculated for each trace independently. Otherwise they are
            calculated for the entire gather.
        use_global : bool, optional, defaults to False
            If `True`, parent survey's mean and std are used, otherwise gather statistics are computed.
        eps : float, optional, defaults to 1e-10
            A constant to be added to the denominator to avoid division by zero.

        Returns
        -------
        self : Gather
            Standardized gather.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        if use_global:
            if not self.survey.has_stats:
                raise ValueError('Global statistics were not calculated, call `Survey.collect_stats` first.')
            mean = self.survey.mean
            std = self.survey.std
        else:
            mean = self._apply_agg_func(func=np.mean, tracewise=tracewise, keepdims=True)
            std = self._apply_agg_func(func=np.std, tracewise=tracewise, keepdims=True)
        self.data = normalization.scale_standard(self.data, mean, std, np.float32(eps))
        return self

    @batch_method(target='threads')
    def scale_maxabs(self, q_min=0, q_max=1, tracewise=True, use_global=False, clip=False, eps=1e-10):
        r"""Scale the gather by its maximum absolute value.

        Maxabs scale of the gather `g` is calculated as:
        :math: `G = \frac{g}{m + eps}`,
        where:
        `m` - the maximum of absolute values of `q_min`-th and `q_max`-th quantiles,
        `eps` - a constant that is added to the denominator to avoid division by zero.

        Quantiles are used to minimize the effect of amplitude outliers on the scaling result. Default 0 and 1
        quantiles represent the minimum and maximum values of the gather respectively and result in usual max-abs
        scaler behavior.

        Notes
        -----
        1. The presence of NaN values in the gather will lead to incorrect behavior of the scaler.
        2. Maxabs scaling is performed inplace.

        Parameters
        ----------
        q_min : float, optional, defaults to 0
            A quantile to be used as a gather minimum during scaling.
        q_max : float, optional, defaults to 1
            A quantile to be used as a gather maximum during scaling.
        tracewise : bool, optional, defaults to True
            If `True`, quantiles are calculated for each trace independently. Otherwise they are calculated for the
            entire gather.
        use_global : bool, optional, defaults to False
            If `True`, parent survey's quantiles are used, otherwise gather quantiles are computed.
        clip : bool, optional, defaults to False
            Whether to clip the scaled gather to the [-1, 1] range.
        eps : float, optional, defaults to 1e-10
            A constant to be added to the denominator to avoid division by zero.

        Returns
        -------
        self : Gather
            Scaled gather.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        min_value, max_value = self.get_quantile([q_min, q_max], tracewise=tracewise, use_global=use_global)
        self.data = normalization.scale_maxabs(self.data, min_value, max_value, clip, np.float32(eps))
        return self

    @batch_method(target='threads')
    def scale_minmax(self, q_min=0, q_max=1, tracewise=True, use_global=False, clip=False, eps=1e-10):
        r"""Linearly scale the gather to a [0, 1] range.

        The transformation of the gather `g` is given by:
        :math:`G=\frac{g - min}{max - min + eps}`
        where:
        `min` and `max` - `q_min`-th and `q_max`-th quantiles respectively,
        `eps` - a constant that is added to the denominator to avoid division by zero.

        Notes
        -----
        1. The presence of NaN values in the gather will lead to incorrect behavior of the scaler.
        2. Minmax scaling is performed inplace.

        Parameters
        ----------
        q_min : float, optional, defaults to 0
            A quantile to be used as a gather minimum during scaling.
        q_max : float, optional, defaults to 1
            A quantile to be used as a gather maximum during scaling.
        tracewise : bool, optional, defaults to True
            If `True`, quantiles are calculated for each trace independently. Otherwise they are calculated for the
            entire gather.
        use_global : bool, optional, defaults to False
            If `True`, parent survey's quantiles are used, otherwise gather quantiles are computed.
        clip : bool, optional, defaults to False
            Whether to clip the scaled gather to the [0, 1] range.
        eps : float, optional, defaults to 1e-10
            A constant to be added to the denominator to avoid division by zero.

        Returns
        -------
        self : Gather
            Scaled gather.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        min_value, max_value = self.get_quantile([q_min, q_max], tracewise=tracewise, use_global=use_global)
        self.data = normalization.scale_minmax(self.data, min_value, max_value, clip, np.float32(eps))
        return self

    #------------------------------------------------------------------------#
    #                    First-breaks processing methods                     #
    #------------------------------------------------------------------------#

    @batch_method(target="threads")
    def pick_to_mask(self, first_breaks_col="FirstBreak", mask_attr="mask"):
        """Convert first break times to a binary mask with the same shape as `gather.data` containing zeros before the
        first arrivals and ones after for each trace.

        Parameters
        ----------
        first_breaks_col : str, optional, defaults to 'FirstBreak'
            A column of `self.headers` that contains first arrival times, measured in milliseconds.
        mask_attr : str, optional, defaults to 'mask'
            Gather attribute to store the mask in.

        Returns
        -------
        self : Gather
            Gather with calculated first breaks mask. Overwrites `mask_attr` attribute if it exists.
        """
        self.validate(required_header_cols=first_breaks_col)
        mask = convert_times_to_mask(times=self[first_breaks_col], sample_rate=self.sample_rate,
                                     mask_length=self.shape[1]).astype(np.int32)
        setattr(self, mask_attr, mask)
        return self

    @batch_method(target='for')
    def mask_to_pick(self, threshold=0.5, first_breaks_col="FirstBreak", mask_attr="mask"):
        """Convert a first break mask into times of first arrivals.

        The mask shape should match the shape of `gather.data`, each its value should represent a probability of
        corresponding index along the trace to follow the first break.

        Notes
        -----
        A detailed description of conversion heuristic used can be found in :func:`~general_utils.convert_mask_to_pick`
        docs.

        Parameters
        ----------
        threshold : float, optional, defaults to 0.5
            A threshold for trace mask value to refer its index to be either pre- or post-first break.
        first_breaks_col : str, optional, defaults to 'FirstBreak'
            Headers column to save first break times to.
        mask_attr : str, optional, defaults to 'mask'
            Gather attribute to get the mask from.

        Returns
        -------
        self : Gather
            A gather with first break times in headers column defined by `first_breaks_col`.

        Raises
        ------
        ValueError
            If an attribute defined by `mask_attr` does not exist.
        """
        mask = getattr(self, mask_attr, None)
        if mask is None:
            raise ValueError(f"Mask attribute given by '{mask_attr}' does not exist")
        self[first_breaks_col] = convert_mask_to_pick(mask, self.sample_rate, threshold)
        return self

    #------------------------------------------------------------------------#
    #                         Gather muting methods                          #
    #------------------------------------------------------------------------#

    @batch_method(target="for", copy_src=False)
    def create_muter(self, mode="first_breaks", **kwargs):
        """Create an instance of :class:`~.Muter` class.

        This method redirects the call into a corresponding `Muter.from_{mode}` classmethod. The created object is
        callable and returns times up to which muting should be performed for given offsets. A detailed description of
        `Muter` instance can be found in :class:`~muting.Muter` docs.

        Parameters
        ----------
        mode : {"points", "file", "first_breaks"}, optional, defaults to "first_breaks"
            Type of `Muter` to create.
        kwargs : misc, optional
            Additional keyword arguments to `Muter.from_{mode}`.

        Returns
        -------
        muter : Muter
            Created muter.

        Raises
        ------
        ValueError
            If given `mode` does not exist.
        """
        builder = getattr(Muter, f"from_{mode}", None)
        if builder is None:
            raise ValueError(f"Unknown mode {mode}")

        if mode == "first_breaks":
            first_breaks_col = kwargs.pop("first_breaks_col", "FirstBreak")
            return builder(offsets=self.offsets, times=self[first_breaks_col], **kwargs)
        return builder(**kwargs)

    @batch_method(target="threads", args_to_unpack="muter")
    def mute(self, muter, fill_value=0):
        """Mute the gather using given `muter`.

        The muting operation is performed by setting gather values above an offset-time boundary defined by `muter` to
        `fill_value`.

        Parameters
        ----------
        muter : Muter
            An object that defines muting times by gather offsets.
        fill_value : float, defaults to 0
            A value to fill the muted part of the gather with.

        Returns
        -------
        self : Gather
            Muted gather.
        """
        self.data = mute_gather(gather_data=self.data, muting_times=muter(self.offsets),
                                sample_rate=self.sample_rate, fill_value=fill_value)
        return self

    #------------------------------------------------------------------------#
    #                     Semblance calculation methods                      #
    #------------------------------------------------------------------------#

    @batch_method(target="threads", copy_src=False)
    def calculate_semblance(self, velocities, win_size=25):
        """Calculate vertical velocity semblance for the gather.

        Notes
        -----
        The gather should be sorted by offset. A detailed description of vertical velocity semblance and its
        computation algorithm can be found in :func:`~semblance.Semblance` docs.

        Examples
        --------
        Calculate semblance for 200 velocities from 2000 to 6000 m/s and a temporal window size of 8 samples:
        >>> gather = gather.sort(by="offset")
        >>> semblance = gather.calculate_semblance(velocities=np.linspace(2000, 6000, 200), win_size=8)

        Parameters
        ----------
        velocities : 1d np.ndarray
            Range of velocity values for which semblance is calculated. Measured in meters/seconds.
        win_size : int, optional, defaults to 25
            Temporal window size used for semblance calculation. The higher the `win_size` is, the smoother the
            resulting semblance will be but to the detriment of small details. Measured in samples.

        Returns
        -------
        semblance : Semblance
            Calculated vertical velocity semblance.

        Raises
        ------
        ValueError
            If the gather is not sorted by offset.
        """
        self.validate(required_sorting="offset")
        return Semblance(gather=self, velocities=velocities, win_size=win_size)

    @batch_method(target="threads", args_to_unpack="stacking_velocity", copy_src=False)
    def calculate_residual_semblance(self, stacking_velocity, n_velocities=140, win_size=25, relative_margin=0.2):
        """Calculate residual vertical velocity semblance for the gather and a chosen stacking velocity.

        Notes
        -----
        The gather should be sorted by offset. A detailed description of residual vertical velocity semblance and its
        computation algorithm can be found in :func:`~semblance.ResidualSemblance` docs.

        Examples
        --------
        Calculate residual semblance for a gather and a stacking velocity, loaded from a file:
        >>> gather = gather.sort(by="offset")
        >>> velocity = StackingVelocity.from_file(velocity_path)
        >>> residual = gather.calculate_residual_semblance(velocity, n_velocities=100, win_size=8)

        Parameters
        ----------
        stacking_velocity : StackingVelocity
            Stacking velocity around which residual semblance is calculated.
        n_velocities : int, optional, defaults to 140
            The number of velocities to compute residual semblance for.
        win_size : int, optional, defaults to 25
            Temporal window size used for semblance calculation. The higher the `win_size` is, the smoother the
            resulting semblance will be but to the detriment of small details. Measured in samples.
        relative_margin : float, optional, defaults to 0.2
            Relative velocity margin, that determines the velocity range for semblance calculation for each time `t` as
            `stacking_velocity(t)` * (1 +- `relative_margin`).

        Returns
        -------
        semblance : ResidualSemblance
            Calculated residual vertical velocity semblance.

        Raises
        ------
        ValueError
            If the gather is not sorted by offset.
        """
        self.validate(required_sorting="offset")
        return ResidualSemblance(gather=self, stacking_velocity=stacking_velocity, n_velocities=n_velocities,
                                 win_size=win_size, relative_margin=relative_margin)

    #------------------------------------------------------------------------#
    #                           Gather corrections                           #
    #------------------------------------------------------------------------#

    @batch_method(target="threads", args_to_unpack="stacking_velocity")
    def apply_nmo(self, stacking_velocity, coords_columns="index"):
        """Perform gather normal moveout correction using given stacking velocity.

        Notes
        -----
        A detailed description of NMO correction can be found in :func:`~correction.apply_nmo` docs.

        Parameters
        ----------
        stacking_velocity : StackingVelocity or VelocityCube
            Stacking velocities to perform NMO correction with. `StackingVelocity` instance is used directly. If
            `VelocityCube` instance is passed, a `StackingVelocity` corresponding to gather coordinates is fetched
            from it.
        coords_columns : None, "index" or 2 element array-like, defaults to "index"
            Header columns to get spatial coordinates of the gather from to fetch `StackingVelocity` from
            `VelocityCube`. See :func:`~Gather.get_coords` for more details.

        Returns
        -------
        self : Gather
            NMO corrected gather.

        Raises
        ------
        ValueError
            If `stacking_velocity` is not a `StackingVelocity` or `VelocityCube` instance.
        """
        if isinstance(stacking_velocity, VelocityCube):
            stacking_velocity = stacking_velocity(*self.get_coords(coords_columns), create_interpolator=False)
        if not isinstance(stacking_velocity, StackingVelocity):
            raise ValueError("Only VelocityCube or StackingVelocity instances can be passed as a stacking_velocity")
        velocities_ms = stacking_velocity(self.times) / 1000  # from m/s to m/ms
        self.data = correction.apply_nmo(self.data, self.times, self.offsets, velocities_ms, self.sample_rate)
        return self

    #------------------------------------------------------------------------#
    #                       General processing methods                       #
    #------------------------------------------------------------------------#

    @batch_method(target="for")
    def sort(self, by):
        """Sort gather `headers` and traces by specified header column.

        Parameters
        ----------
        by : str
            `headers` column name to sort the gather by.

        Returns
        -------
        self : Gather
            Gather sorted by `by` column. Sets `sort_by` attribute to `by`.

        Raises
        ------
        TypeError
            If `by` is not str.
        ValueError
            If `by` column was not loaded in `headers`.
        """
        if not isinstance(by, str):
            raise TypeError('`by` should be str, not {}'.format(type(by)))
        self.validate(required_header_cols=by)
        order = np.argsort(self.headers[by].values, kind='stable')
        self.sort_by = by
        self.data = self.data[order]
        self.headers = self.headers.iloc[order]
        return self

    @batch_method(target="for")
    def get_central_cdp(self):
        """Get a central CDP gather from a supergather.

        A supergather has `SUPERGATHER_INLINE_3D` and `SUPERGATHER_CROSSLINE_3D` headers columns, whose values are
        equal to values in `INLINE_3D` and `CROSSLINE_3D` only for traces from the central CDP gather. Read more about
        supergather generation in :func:`~Survey.generate_supergathers` docs.

        Returns
        -------
        self : Gather
            `self` with only traces from the central CDP gather kept. Updates `self.headers` and `self.data` inplace.

        Raises
        ------
        ValueError
            If any of the `INLINE_3D`, `CROSSLINE_3D`, `SUPERGATHER_INLINE_3D` or `SUPERGATHER_CROSSLINE_3D` columns
            are not in `headers`.
        """
        self.validate(required_header_cols=["INLINE_3D", "SUPERGATHER_INLINE_3D",
                                            "CROSSLINE_3D", "SUPERGATHER_CROSSLINE_3D"])
        headers = self.headers.reset_index()
        mask = ((headers["SUPERGATHER_INLINE_3D"] == headers["INLINE_3D"]) &
                (headers["SUPERGATHER_CROSSLINE_3D"] == headers["CROSSLINE_3D"])).values
        self.headers = self.headers.loc[mask]
        self.data = self.data[mask]
        return self

    @batch_method(target="for")
    def stack(self):
        """Stack a gather by calculating mean value of all non-nan amplitudes for each time over the offset axis.

        The gather after stacking contains only one trace. Its `headers` is indexed by `INLINE_3D` and `CROSSLINE_3D`
        and has a single `TRACE_SEQUENCE_FILE` header with a value of 1.

        Notes
        -----
        Only a CDP gather indexed by `INLINE_3D` and `CROSSLINE_3D` can be stacked.

        Raises
        ------
        ValueError
            If the gather is not indexed by `INLINE_3D` and `CROSSLINE_3D` or traces from multiple CDP gathers are
            being stacked
        """
        line_cols = ["INLINE_3D", "CROSSLINE_3D"]
        self.validate(required_header_cols=line_cols)
        headers = self.headers.reset_index()[line_cols].drop_duplicates()
        if len(headers) != 1:
            raise ValueError("Only a single CDP gather can be stacked")
        self.headers = headers.set_index(line_cols)
        self.headers["TRACE_SEQUENCE_FILE"] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.data = np.nanmean(self.data, axis=0, keepdims=True)
        self.data = np.nan_to_num(self.data)
        return self

    @batch_method(target='for')
    def crop(self, crop_rule, crop_size):
        return CroppedGazer(self, crop_rule, crop_size)

    #------------------------------------------------------------------------#
    #                         Visualization methods                          #
    #------------------------------------------------------------------------#

    @batch_method(target="for", copy_src=False)
    def plot(self, figsize=(10, 7), **kwargs):
        """Plot gather traces.

        Parameters
        ----------
        figsize : tuple, optional, defaults to (10, 7)
            Output plot size.
        kwargs : misc, optional
            Additional keyword arguments to `matplotlib.pyplot.imshow`.

        Returns
        -------
        self : Gather
            Gather unchanged.
        """
        vmin, vmax = self.get_quantile([0.1, 0.9])
        default_kwargs = {
            'cmap': 'gray',
            'vmin': vmin,
            'vmax': vmax,
            'aspect': 'auto',
        }
        default_kwargs.update(kwargs)
        plt.figure(figsize=figsize)
        plt.imshow(self.data.T, **default_kwargs)
        plt.show()
        return self
