"""AVO"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import aggregate_by_bins_numba
from .. import SeismicDataset
from ..utils import to_list, save_figure


class AmplitudeOffsetDistribution:
    def __init__(self, headers, agg_stats, indexed_by, bins_bounds, name):
        self.headers = headers
        self.agg_stats = agg_stats
        self.indexed_by = indexed_by
        self.bins_bounds = bins_bounds
        self.name = name
        self.data = None
        self.gather_bounds = None

        self.bins_mean = np.nanmean(agg_stats, axis=0)
        self.bins_approx = None

        # Metrics
        self.metrics = {}

    @classmethod
    def from_survey(cls, survey, avo_column, bin_size, indexed_by=None, name=None):
        headers = survey.headers.reset_index()
        if "offset" not in headers:
            raise ValueError("Missing offset header")
        name = name if name is not None else survey.name

        data = headers[[avo_column, "offset"]].to_numpy()
        gather_bounds = None
        if indexed_by is None:
            # Use index from survey by default and avoid further data reindex
            gather_bounds = np.where(~survey.headers.index.duplicated())[0]
            gather_bounds = np.array(tuple(zip(gather_bounds[:-1], gather_bounds[1:]-1)))
            indexed_by = survey.indexed_by
        else:
            gather_bounds, groups_ix = cls._reindex(headers=headers, new_index=indexed_by)
            data = data[groups_ix]

        bins_bounds = cls._get_bounds_by_bin_size(bin_size, headers["offset"].max())
        agg_stats = cls._compute_in_bins(data=data, bins_bounds=bins_bounds, gather_bounds=gather_bounds)

        self = cls(headers, agg_stats, indexed_by, bins_bounds, name)
        self.data = data
        self.gather_bounds = gather_bounds
        return self

    @classmethod
    def from_file(cls):
        return cls

    def regroup(self, bin_size=None, indexed_by=None, name=None):
        """Create new instance of AmplitudeOffsetDistribution with different bin_size or indexed by different header"""
        data = self.data
        bins_bounds = self.bins_bounds
        gather_bounds = self.gather_bounds
        name = name if name is not None else self.name

        if bin_size is not None:
            bins_bounds = self._get_bounds_by_bin_size(bin_size, self.headers["offset"].max())

        if indexed_by is not None and set(to_list(indexed_by)) != set(self.indexed_by):
            gather_bounds, groups_ix = self._reindex(headers=self.headers, new_index=indexed_by)
            data = data[groups_ix]

        agg_stats = self._compute_in_bins(data=data, bins_bounds=bins_bounds, gather_bounds=gather_bounds)

        new_self = type(self)(self.headers.copy(deep=False), indexed_by, bins_bounds, agg_stats, name)
        new_self.data = data
        new_self.gather_bounds = gather_bounds
        return new_self

    @staticmethod
    def _reindex(headers, new_index):
        new_index = to_list(new_index)
        headers_grouper = headers.groupby(new_index).grouper
        groups_ix = headers_grouper.result_ilocs()
        gather_bounds = np.cumsum([0, *headers_grouper.size().to_numpy()])
        gather_bounds = np.array(tuple(zip(gather_bounds[:-1], gather_bounds[1:]-1)))
        return gather_bounds, groups_ix

    @staticmethod
    def _get_bounds_by_bin_size(bin_size, max_offset):
        if isinstance(bin_size, (int, np.integer)):
            return np.arange(0, max_offset+bin_size, bin_size)
        return np.cumsum([0, *bin_size])

    @staticmethod
    def _compute_in_bins(data, bins_bounds, gather_bounds):
        avo_stats, offsets = data.T
        agg_stats = aggregate_by_bins_numba(avo_stats, offsets, bins_bounds, gather_bounds)
        agg_stats[agg_stats == 0] = np.nan
        return agg_stats

    def qc(self, names, pol_degree=3):

        metrics_dict = {
            "std": lambda : np.mean(np.nanstd(self.agg_stats, axis=1)),
            "corr": self._calculate_corr
        }

        for name in to_list(names):
            metric_func = metrics_dict.get(name)
            if metric_func is None:
                raise ValueError("")
            # TODO: figure out how to pass kwargs to metric func normally
            kwargs = {"pol_degree" : pol_degree} if name == "corr" else {}
            self.metrics[name] = metric_func(**kwargs)

    def _calculate_corr(self, pol_degree):
        not_nans = ~np.isnan(self.bins_mean)
        bounds = self.bins_bounds[:-1][not_nans]
        means = self.bins_mean[not_nans]
        poly = np.polyfit(bounds, means, deg=pol_degree)
        self.bins_approx = np.polyval(poly, self.bins_bounds[:-1])
        return np.corrcoef(means, np.polyval(poly, bounds))[0][1]

    def plot(self, show_qc=False, show_poly=False, title=None, figsize=(12, 7), dot_size=2, avg_size=8, dpi=100,
             save_to=None):
        fig, ax = plt.subplots(figsize=figsize)
        indices = np.tile(self.bins_bounds[:-1], self.agg_stats.shape[0])
        ax.plot(indices, self.agg_stats.ravel(), 'o', markersize=dot_size)
        ax.plot(self.bins_bounds[:-1], self.bins_mean, 'v', color='r', markersize=avg_size)
        if show_poly:
            if self.bins_approx is None:
                raise ValueError("Calculate QC for `corr` before using `show_poly`")
            ax.plot(self.bins_bounds[:-1], self.bins_approx, '--', c='g', zorder=3)
        ax.grid()
        self._finalize_plot(fig, ax, show_qc, title, save_to, dpi)

    def plot_std(self, *args, align=False, title=None, figsize=(12, 7), dpi=100,
                 save_to=None):
        if self.name is None:
            raise ValueError("self.name must be specified")

        data = self.agg_stats.ravel()
        indices = np.tile(self.bins_bounds[:-1], self.agg_stats.shape[0])
        names = [self.name] * len(indices)
        if args:
            for other in args:
                if other.name is None:
                    raise ValueError("`name` attribute must be specified for all AVOs")
                if np.any(self.bins_bounds != other.bins_bounds):
                    raise ValueError("All AVOs must have same bin_size")
                if np.any(self.indexed_by != other.indexed_by):
                    raise ValueError("All AVOs must be indexed by the same header")

                other_data = other.agg_stats.ravel()
                data = np.append(data, other_data)
                indices = np.append(indices, np.tile(self.bins_bounds[:-1], other.agg_stats.shape[0]))
                names += [other.name] * len(other_data)

        if align:
            # Align all avo stats to the same mean
            global_mean = np.nanmean(data)
            ixs = np.append([0], np.where(np.diff(indices) < 0))
            for start, end in zip(ixs[:-1], ixs[1:]):
                data[start: end] += global_mean - np.nanmean(data[start: end])

        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(x=indices, y=data, hue=names, ci="sd", ax=ax)

        self._finalize_plot(fig, ax, False, title, save_to, dpi)
        sns.set_style("ticks")

    def _finalize_plot(self, fig, ax, show_qc, title, save_to, dpi):
        ax.set_xlabel('Offset')
        ax.set_ylabel('Amplitude')
        if show_qc:
            title = "" if title is None else title
            for name, value in self.metrics.items():
                sep = "\n" if title else ""
                title = title + sep + f"{name} : {value:.4}"
        ax.set_title(title)
        if save_to is not None:
            save_figure(fig, save_to, dpi=dpi)

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
