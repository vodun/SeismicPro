"""AVO"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import aggregate_by_bins_numba
from .. import SeismicDataset
from ..utils import to_list, save_figure


class AmplitudeOffsetDistribution:
    def __init__(self, stats_df, avo_column, headers, indexed_by, name):
        self.stats_df = stats_df
        self.avo_column = avo_column
        self.headers = headers
        self.indexed_by = indexed_by
        self.name = name

        self.bins_df = headers.groupby("bin", as_index=False)[avo_column].mean()

        # Metrics
        self.metrics = {}

    @classmethod
    def from_survey(cls, survey, avo_column, bin_size, indexed_by=None, name=None):
        name = name if name is not None else survey.name
        indexed_by = indexed_by if indexed_by is not None else survey.indexed_by
        return cls.from_headers(headers=survey.headers, avo_column=avo_column, bin_size=bin_size, indexed_by=indexed_by,
                                name=name)

    def regroup(self, bin_size=None, indexed_by=None, name=None):
        """Create new instance of AmplitudeOffsetDistribution with different bin_size or indexed by different header"""
        name = name if name is not None else self.name
        bin_size = self.bin_bounds if bin_size is None else bin_size
        indexed_by = self.indexed_by if indexed_by is None else indexed_by
        return cls.from_headers(headers=self.headers, avo_column=self.avo_column, bin_size=bin_size,
                                indexed_by=indexed_by, name=name)

    @classmethod
    def from_headers(cls, headers, avo_column, bin_size, indexed_by, name=None):
        if "offset" not in headers:
            raise ValueError("Missing offset header")
        headers = headers.copy(deep=False)
        bin_bounds = cls._get_bin_bounds(bin_size, headers["offset"].max())
        headers["bin"] = bin_bounds[np.searchsorted(bin_bounds, headers["offset"], side='right')]
        indexed_by = to_list(indexed_by)

        stats_df = headers.groupby([*indexed_by, "bin"], as_index=False)[avo_column].mean()
        return cls(stats_df=stats_df, avo_column=avo_column, headers=headers, indexed_by=indexed_by, name=name)

    @classmethod
    def from_file(cls):
        return cls

    @staticmethod
    def _get_bin_bounds(bin_size, max_offset):
        if isinstance(bin_size, (int, np.integer)):
            return np.arange(0, max_offset+bin_size, bin_size)
        return np.cumsum([0, *bin_size])

    def qc(self, names=None, pol_degree=3):

        metrics_dict = {
            "std": self._calculate_std,
            "corr": self._calculate_corr
        }
        names = names if names is not None else ["std", "corr"]
        for name in to_list(names):
            metric_func = metrics_dict.get(name)
            if metric_func is None:
                raise ValueError("")
            # TODO: figure out how to pass kwargs to metric func normally
            kwargs = {"pol_degree" : pol_degree} if name == "corr" else {}
            self.metrics[name] = metric_func(**kwargs)

    def _calculate_std(self):
        return self.stats_df.groupby("bin")[self.avo_column].apply(np.nanstd).mean()

    def _calculate_corr(self, pol_degree):
        mask = ~self.bins_df[self.avo_column].isna()
        not_nan_bins = self.bins_df[mask]
        poly = np.polyfit(not_nan_bins["bin"], not_nan_bins[self.avo_column], deg=pol_degree)

        bins_approx = np.full(len(mask), np.nan)
        bins_approx[mask] = np.polyval(poly, not_nan_bins["bin"])
        self.bins_df["bins_approx"] = bins_approx
        return np.corrcoef(not_nan_bins[self.avo_column], bins_approx[mask])[0][1]

    def plot(self, show_qc=False, show_poly=False, title=None, figsize=(12, 7), dot_size=2, avg_size=50, dpi=100,
             save_to=None):
        fig, ax = plt.subplots(figsize=figsize)
        self.stats_df.plot(x='bin', y=self.avo_column, kind='scatter', ax=ax, s=dot_size)
        self.bins_df.plot(x='bin', y=self.avo_column, kind='scatter', ax=ax, s=avg_size, marker='v', color='r',
                            grid=True)

        if show_poly:
            if "bins_approx" not in self.bins_df:
                raise ValueError("Calculate QC for `corr` before using `show_poly`")
            ax.plot(self.bins_df["bin"], self.bins_df["bins_approx"], '--', c='g', zorder=3)
        self._finalize_plot(fig, ax, show_qc, title, save_to, dpi)

    def plot_std(self, *args, align=False, title=None, figsize=(12, 7), dpi=100, save_to=None):
        avos = [self, *args]
        if align:
            global_mean = np.nanmean([avo.stats_df[avo.avo_column] for avo in avos])

        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=figsize)
        for avo in avos:
            stats = avo.stats_df.copy(deep=False)
            # TODO: Optimize with single if align
            if align:
                stats[avo.avo_column] = stats[avo.avo_column] + global_mean - np.nanmean(stats[avo.avo_column])
            # TODO: Do we need checks on the same bin_size and indexed_by?
            sns.lineplot(data=stats, x="bin", y=avo.avo_column, ci="sd", ax=ax, label=avo.name)
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
