"""The file contains AVO dataset."""
import numpy as np
import pandas as pd

from seismicpro.src import SeismicDataset, SeismicBatch
from .utils import load_horizon, update_avo_params

class AvoDataset(SeismicDataset):
    """ Child of SeismicDataset. Used for constructing AVO distribution.

    Attributes
    ----------
    horizon: array-like
        Horizon values.
    horizon_name: str
        Path to used horizon.
    """

    def __init__(self, index, batch_class=SeismicBatch, preloaded=None, *args, **kwargs):
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
        self.horizon = None
        self.horizon_name = None

    def find_avo_distribution(self, batch, src, class_size, window=None, horizon_path=None, horizon_window=10,
                              method='rms', container_name='avo_classes'):
        r""" Calculate AVO distribution for all dataset and save result to dataset variable named `container_name`.

        Parameters
        ----------
        batch : SeismicBatch or B() named expression.
            Current batch from pipeline.
        src : str or array-like
            Component with shot gathers or list of components' name.
        class_size : int, array-like or 'offset'
            Lenght of one class or lenght of each class if iterable.
            If 'offset' each class will contain traces from the same offset.
        window : array-like with size 2 or None, optional
            The interval in ms where to constract AVO distribution in the following order [from ms, to ms].
        horizon_path : str or None, optional
            Path to horizon. If not None, AVO will be calculated based on horizon
            data even if `window` is specified.
        horizon_window : int or array-like with length 2, optional, default 10
            If int is given, resulted window will have width along the horizon equal to given number with
            middle in particular horizon value.
            If array with length 2 is given, first value determine the difference between horizon value and lowerbound.
            Second value determine the difference between horizon and upperbound.
            For example, horizon equal to 100, and `horizon_window` is [10, 10], thus the resulted widnow
            will be: [90, 110]. Also, bounders are allows to be negative, thus given `horizon_window` as [-10, 20]
            one will receive [110, 120] with same horizon  value.
        method : 'rms', 'abs' or callable, default 'rms'
            There are two basic types of aggregation amplitudes inside one class.
            'rms' - This method calculate RMS value for each trace and mean all rms values in one class:

                $$ rms = \sum_{i=0}^K {{\sqrt{\sum_{j=0}^N f_{ij}**2}/N}/K $$
                where, K - number of traces in one class.
                N - length of window.
                f - amplitude of traces.

            'abs' - Use sum module of amplitudes of traces for each class:

                $$abs = \sum_{i=0}^K \sum{j=0}^N |f_{ij}|

        contaier_name : str, optional
            Name of the `AvoDataset` attribute to store a dict with estimated AVO. This variable has shape:
            (Number of seismograms in dataset, Number of classes). Each value in container corresponds to
            the calculated rms or abs for a particular seismogram belonging to a particular class.

        Raises
        ------
        ValueError : If trases is not sorted by `offset`.
        ValueError : If `window` is not array-like.
        ValueError : If `field_type` is not 'same' or 'diff'.

        Note
        ----
        1. This function works properly only with CDP index name with sorting by `offset`.
        2. Dictoinary with estimated AVO classes can be obtained from pipeline using `D(container_name)`.
        3. Parameter `horizon_path` have a priority over `window`.
        """
        src = [src] if isinstance(src, str) else src
        container_name = [container_name] if isinstance(container_name, str) else container_name
        window_list = [None] * len(src)

        if horizon_path is not None:
            if isinstance(horizon_window, int):
                horizon_window = [horizon_window // 2, horizon_window // 2]

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
                    raise ValueError(f"Wrong type of `horizon_path` variable. Should be array-like or str"
                                        f", not {type(horizon_path)}")
        elif window is not None:
            if not isinstance(window, (list, tuple, np.ndarray)):
                raise ValueError(f"Wrong type of `window` variable. Should be array-like with size 2"
                                    f", not {type(window)}")
            for i, comp in enumerate(src):
                time_step = np.diff(batch.meta[comp]['samples'][:2])[0]
                window_list[i] = np.round(np.array(window) / time_step).astype(np.int32)
        else:
            raise ValueError("One of variables `horizon_path` or `window` should be determined.")

        if isinstance(class_size, (tuple, list, np.ndarray)):
            class_size = np.cumsum(class_size)
        elif class_size == 'offset':
            class_size = 1

        if method == 'rms':
            # First, calculate average RMS for each trace in subfield. The average RMS value
            # is used to calculate the AVO.
            method = lambda subfield: np.nanmean(np.nanmean(subfield**2, axis=1)**.5)
        elif method == 'abs':
            # Calculate the average value for the module of subfield amplitudes.
            method = lambda subfield: np.nanmean(np.abs(subfield))
        elif not callable(method):
            raise ValueError(f"`method` as src should be either 'rms' or 'abs' or callable, not {method}.")

        for comp, container, wind in zip(src, container_name, window_list):
            sorting = batch.meta[comp]['sorting']
            if sorting != 'offset':
                ValueError(f"Wrong sorting type for component {comp}. Should be 'offset' but given {sorting}.")

            params = getattr(self, container, None)

            if params is not None:
                storage_size = params.shape[1]
            else:
                storage_size = np.ceil(np.max(self.index.get_df()['offset']) / class_size).astype(int)
            params = update_avo_params(params=params, batch=batch, component=comp, class_size=class_size,
                                       storage_size=storage_size, window=wind, horizon_window=horizon_window,
                                       horizon=self.horizon, calc_method=method)
            setattr(self, container, params)
