"""The file contains AVO dataset."""
import sys

import numpy as np
import pandas as pd

# pylint: disable=wrong-import-position
from seismicpro.src import SeismicDataset, SeismicBatch
from .utils import load_horizon, update_avo_params


class AvoDataset(SeismicDataset):
    """ Child of SeismicDataset. Used for constructing AVO distribution.

    Attributes
    ----------
    horizon: array-like
        Horizon values.
    horizon_name: str
        Path to the used horizon.
    """

    def __init__(self, index, batch_class=SeismicBatch, preloaded=None, *args, **kwargs):
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
        self.horizon = None
        self.horizon_name = None

    def find_avo_distribution(self, batch, src, bin_size, window=None, horizon_path=None, horizon_window=10, # pylint: disable=too-many-statements
                              method='rms', container_name='avo_bins'):
        r""" Calculate AVO distribution for the whole dataset and save the result to a dataset variable named
        `container_name`.

        Parameters
        ----------
        batch : SeismicBatch or B() named expression.
            Current batch from pipeline.
        src : str or array-like
            Component with shot gathers or list of components' names.
        bin_size : int, array-like or 'offset'
            Length of one bin or length of each bin if iterable.
            If 'offset' each bin will contain traces from the same offset. This option is equivalent to `bin_size` = 1.
        window : array-like with size 2 or None, optional
           Time frame to calculate AVO distribution in. Should be in the format [from ms, to ms].
        horizon_path : str or None, optional
            Path to the horizon. If not None, AVO will be calculated based on horizon data even if `window`
            is specified.
        horizon_window : int or array-like with length 2, optional, default 10
            If int is given, the resulted window will have a width along the horizon equal to `horizon_window * 2 + 1`
            with particular horizon value in the middle.
            If an array with length 2 is given, the first value determines the difference between the horizon value and
            the lower bound. The second value determines the difference between the horizon and upper bound.
            For example, horizon equal to 100, and `horizon_window` is [10, 10], thus the resulted window will be:
            [90, 110]. Also, bounders might be negative, thus given `horizon_window` as [-10, 20] one will receive
            [110, 120] with the same horizon value.
            Measures in ms.
        method : 'rms', 'abs' or callable, default 'rms'
            There are two basic types of aggregation amplitudes inside one bin.
            'rms' - This method calculates RMS value for each trace and averages all RMS values in one bin:

                $$ rms = \sum_{i=0}^K {{\sqrt{\sum_{j=0}^N f_{ij}**2}/N}/K $$
                where, K - number of traces in one bin.
                N - length of the window.
                f - amplitude of traces.

            'abs' - Calculates sum of absolute values of amplitudes of traces in each bin:

                $$abs = \sum_{i=0}^K \sum{j=0}^N |f_{ij}|

        container_name : str, optional
            Name of the `AvoDataset` attribute to store a 2d-array with estimated AVO. This variable has a shape:
            (Number of seismograms in the dataset, Number of bins). Each value in the container corresponds to the
            calculated rms or abs for a particular seismogram belonging to a particular bin.

        Raises
        ------
        ValueError : If traces are not sorted by `offset`.
        ValueError : If `window` is not array-like.
        ValueError : If `field_type` is not 'same' or 'diff'.

        Note
        ----
        1. This function works properly only with CDP index name with sorting by `offset`.
        2. Dictionary with estimated AVO bins can be obtained from the pipeline using `D(container_name)`.
        3. Parameter `horizon_path` have a priority over `window`.
        """
        src = [src] if isinstance(src, str) else src
        container_name = [container_name] if isinstance(container_name, str) else container_name
        window_list = [None] * len(src)

        if horizon_path is not None:
            if isinstance(horizon_window, int):
                horizon_window = (horizon_window, horizon_window)

            # Check whether is the same horizon or new one.
            if isinstance(horizon_path, str):
                if_new_horizon = horizon_path != self.horizon_name
            else:
                if_new_horizon = np.all(np.array(horizon_path != self.horison_name))

            if self.horizon is None or if_new_horizon:
                self.horizon_name = horizon_path
                col_names = batch.index.get_df().columns
                for title in ['offset', 'CROSSLINE_3D', 'INLINE_3D']:
                    if not title in col_names:
                        raise ValueError("Missing extra_header: {0} in Index dataframe. Add {0} as `extra_headers`"
                                         "to Index. Your list of headers is the following {1}".format(title,
                                                                                                      col_names))

                if isinstance(horizon_path, str):
                    self.horizon = load_horizon(horizon_path)
                elif isinstance(horizon_path, (list, tuple, np.ndarray)):
                    self.horizon = pd.DataFrame(horizon_path, columns=['INLINE_3D', 'CROSSLINE_3D', 'time'])
                else:
                    raise ValueError("Wrong type of `horizon_path` variable. Should be array-like or str"
                                     ", not {}".format(type(horizon_path)))
        elif window is not None:
            if not isinstance(window, (list, tuple, np.ndarray)):
                raise ValueError("Wrong type of `window` variable. Should be array-like with size 2"
                                 ", not {}".format(type(window)))
            for i, comp in enumerate(src):
                time_step = np.diff(batch.meta[comp]['samples'][:2])[0]
                window_list[i] = np.round(np.array(window) / time_step).astype(np.int32)
        else:
            raise ValueError("One of variables `horizon_path` or `window` should be determined.")

        if isinstance(bin_size, (tuple, list, np.ndarray)):
            bin_size = np.cumsum(bin_size)
        elif bin_size == 'offset':
            bin_size = 1

        if method == 'rms':
            # First, calculate average RMS for each trace in subfield. The average RMS value
            # is used to calculate the AVO.
            method = lambda subfield: np.nanmean(np.nanmean(subfield**2, axis=1)**.5)
        elif method == 'abs':
            # Calculate the average value for the module of subfield amplitudes.
            method = lambda subfield: np.nanmean(np.abs(subfield))
        elif not callable(method):
            raise ValueError("`method` as src should be either 'rms' or 'abs' or callable, not {}.".format(method))

        for comp, container, wind in zip(src, container_name, window_list):
            sorting = batch.meta[comp]['sorting']
            if sorting != 'offset':
                ValueError("Wrong sorting type for component {}. Should be 'offset' but given {}.".format(comp,
                                                                                                          sorting))

            params = getattr(self, container, None)
            if params is not None:
                storage_size = params.shape[1]
            else:
                if isinstance(bin_size, (tuple, list, np.ndarray)):
                    storage_size = len(bin_size)
                else:
                    storage_size = np.ceil(np.max(self.index.get_df()['offset']) / bin_size).astype(int)
            params = update_avo_params(params=params, batch=batch, component=comp, bin_size=bin_size,
                                       storage_size=storage_size, window=wind, horizon_window=horizon_window,
                                       horizon=self.horizon, calc_method=method)
            setattr(self, container, params)
