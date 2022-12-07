import os

import numpy as np
import matplotlib.pyplot as plt


from .utils import aggregate_by_bins_numba
from .. import SeismicDataset
from ..utils import to_list

# """TODO: Save values, split them by offset based on given bins. be able to regroup them by any given index (coords)"""
class AmplitudeOffsetDistribution:
    def __init__(self, survey, avo_column, bin_size, indexed_by=None):
        self.headers = survey.headers.reset_index().copy(deep=False)
        if "offset" not in self.headers:
            raise ValueError("Missing offset header")
        self.max_offset = self.headers["offset"].max()
        self.avo_column = avo_column

        self._indexed_by = None
        self.gather_bounds = None
        self.data = self.headers[[self.avo_column, "offset"]].to_numpy()
        self.indexed_by = indexed_by or survey.indexed_by

        self.bin_bounds = None
        self.agg_stats = None
        self.combine_to_gathers(bin_size=bin_size)

    @property
    def indexed_by(self):
        return self._indexed_by

    @indexed_by.setter
    def indexed_by(self, columns):
        columns = to_list(columns)
        headers_grouper = self.headers.groupby(columns).grouper
        groups_ix = headers_grouper.result_ilocs()
        gathers_bounds = np.cumsum([0, *headers_grouper.size().to_numpy()])

        self.data = self.data[groups_ix]
        self.gather_bounds = np.array(tuple(zip(gathers_bounds[:-1], gathers_bounds[1:]-1)))
        self._indexed_by = columns

    def combine_to_gathers(self, bin_size=None, indexed_by=None):
        if isinstance(bin_size, (int, np.integer)):
            self.bin_bounds = np.arange(0, self.max_offset+bin_size, bin_size)
        elif hasattr(bin_size, "__iter__"):
            self.bin_bounds = np.cumsum([0, *bin_size])

        if indexed_by is not None and set(to_list(indexed_by)) != set(self.indexed_by):
            self.indexed_by = indexed_by

        avo_stats, offsets = self.headers[[self.avo_column, "offset"]].to_numpy().T
        self.agg_stats = aggregate_by_bins_numba(avo_stats, offsets, self.bin_bounds, self.gather_bounds)

    def plot(self, title=None, figsize=(12, 7), dot_size=3, avg_size=8, dpi=100, save_to=None):
        indices = np.tile(self.bin_bounds, self.agg_stats.shape[0])

        bins_mean = np.nanmean(self.agg_stats, axis=0)
        plt.figure(figsize=figsize)
        plt.plot(indices, self.agg_stats.ravel(), '.', markersize=3)
        plt.plot(self.bin_bounds, bins_mean, 'v', color='r', markersize=8)

        plt.ylabel('Amplitude')
        plt.xlabel('Offset')
        if title:
            plt.title(title)
        plt.grid()

        if save_to is not None:
            plt.savefig(save_to, dpi=dpi)
        plt.show()

    # def plot_std(self, *args):






    # @classmethod
    # def from_survey(cls, survey, bins, combine=False, method_kwargs=None, pipeline_kwargs=None):
    #     method_kwargs = {} if method_kwargs is None else method_kwargs
    #     pipeline_kwargs = {} if pipeline_kwargs is None else pipeline_kwargs

    #     name = survey.name
    #     ds = SeismicDataset(survey)
    #     _ = (ds.p
    #            .load(src=name, combine=combine)
    #            .calculate_avo(src=name, **method_kwargs)
    #            .store_headers_to_survey(src=name)
    #     ).run(**pipeline_kwargs)

    #     return cls(survey[["avo_stats"]], survey["offset"], bins)
