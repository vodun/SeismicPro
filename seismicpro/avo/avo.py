import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import aggregate_by_bins_numba
from .. import SeismicDataset
from ..utils import to_list


class AmplitudeOffsetDistribution:
    def __init__(self, survey, avo_column, bin_size, indexed_by=None, name=None):
        self.headers = survey.headers.reset_index().copy(deep=False)
        if "offset" not in self.headers:
            raise ValueError("Missing offset header")
        self.max_offset = self.headers["offset"].max()
        self.avo_column = avo_column

        self.data = self.headers[[self.avo_column, "offset"]].to_numpy()
        self.indexed_by = None
        if indexed_by is None:
            # Use index from survey by default and avoid further data reindex
            gather_bounds = np.where(~survey.headers.index.duplicated())[0]
            self.gather_bounds = np.array(tuple(zip(gather_bounds[:-1], gather_bounds[1:]-1)))
            self.indexed_by = survey.indexed_by

        self.name = name if name is not None else survey.name
        self.bin_bounds = None
        self.agg_stats = None
        self.regroup(bin_size=bin_size, indexed_by=indexed_by or self.indexed_by)

    def regroup(self, bin_size=None, indexed_by=None):
        if bin_size is not None:
            if isinstance(bin_size, (int, np.integer)):
                self.bin_bounds = np.arange(0, self.max_offset+bin_size, bin_size)
            else:
                self.bin_bounds = np.cumsum([0, *bin_size])

        if self.bin_bounds is None:
            raise ValueError("`bin_size` is missed.")

        if indexed_by is not None and set(to_list(indexed_by)) != set(self.indexed_by):
            self.reindex(new_index=indexed_by)

        avo_stats, offsets = self.data.T
        agg_stats = aggregate_by_bins_numba(avo_stats, offsets, self.bin_bounds, self.gather_bounds)
        agg_stats[agg_stats == 0] = np.nan
        self.agg_stats = agg_stats

    def reindex(self, new_index):
        new_index = to_list(new_index)
        headers_grouper = self.headers.groupby(new_index).grouper
        groups_ix = headers_grouper.result_ilocs()
        gathers_bounds = np.cumsum([0, *headers_grouper.size().to_numpy()])

        self.data = self.data[groups_ix]
        self.gather_bounds = np.array(tuple(zip(gathers_bounds[:-1], gathers_bounds[1:]-1)))
        self.indexed_by = new_index

    def qc(self, mode):
        return self

    def plot(self, title=None, figsize=(12, 7), dot_size=2, avg_size=8, dpi=100, save_to=None):
        indices = np.tile(self.bin_bounds[:-1], self.agg_stats.shape[0])

        bins_mean = np.nanmean(self.agg_stats, axis=0)
        fig = plt.figure(figsize=figsize)
        plt.plot(indices, self.agg_stats.ravel(), 'o', markersize=dot_size)
        plt.plot(self.bin_bounds[:-1], bins_mean, 'v', color='r', markersize=avg_size)

        plt.grid()
        plt.xlabel('Offset')
        plt.ylabel('Amplitude')
        plt.title(title)

        if save_to is not None:
            save_figure(fig, save_to, dpi=dpi)
        plt.show()

    def plot_std(self, *args, align=False, title=None, figsize=(12, 7), save_to=None):
        if self.name is None:
            raise ValueError("self.name must be specified")

        data = self.agg_stats.ravel()
        indices = np.tile(self.bin_bounds[:-1], self.agg_stats.shape[0])
        names = [self.name] * len(indices)
        if args:
            for ix, other in enumerate(args):
                if other.name is None:
                    raise ValueError("`name` attribute must be specified for all AVOs")
                if np.any(self.bin_bounds != other.bin_bounds):
                    raise ValueError("All AVOs must have same bin_size")
                if np.any(self.indexed_by != other.indexed_by):
                    raise ValueError("All AVOs must be indexed by the same header")

                other_data = other.agg_stats.ravel()
                data = np.append(data, other_data)
                indices = np.append(indices, np.tile(self.bin_bounds[:-1], other.agg_stats.shape[0]))
                names += [other.name] * len(other_data)

        if align:
            # Align all avo stats to the same mean
            global_mean = np.nanmean(data)
            ixs = np.append([0], np.where(np.diff(indices) < 0))
            for start, end in zip(ixs[:-1], ixs[1:]):
                data[start: end] += global_mean - np.nanmean(data[start: end])

        sns.set_style("darkgrid")
        fig = plt.figure(figsize=figsize)
        sns.lineplot(x=indices, y=data, hue=names, ci="sd")
        plt.xlabel('Offset')
        plt.ylabel('Amplitude')
        plt.title(title)
        if save_to is not None:
            save_figure(fig, save_to, dpi=dpi)
        plt.show()
        sns.set_style("ticks")


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
