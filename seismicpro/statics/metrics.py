from functools import partial

import numpy as np
from numba import njit
from scipy.optimize import curve_fit

from ..metrics import Metric


class TravelTimeMetric(Metric):
    def __init__(self, nsm, survey_list, coords_cols, **kwargs):
        super().__init__(**kwargs)
        self.nsm = nsm
        self.survey_list = [survey.reindex(coords_cols) for survey in survey_list]

    def plot_on_click(self, coords, ax, sort_by=None, **kwargs):
        survey = [survey for survey in self.survey_list if coords in survey.indices][0]
        gather = survey.get_gather(coords, copy_headers=True)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        is_uphole = self.nsm.is_uphole
        if is_uphole is None:
            loaded_headers = set(survey.headers.columns) | set(survey.headers.index.names)
            is_uphole = "SourceDepth" in loaded_headers
        shots_depths = gather["SourceDepth"] if is_uphole else 0
        fb_pred = self.nsm.estimate_traveltimes(gather[["SourceX", "SourceY"]], gather[["GroupX", "GroupY"]],
                                                shots_depths=shots_depths)
        gather["PredictedFirstBreaks"] = fb_pred
        gather.plot(ax=ax, event_headers=[self.nsm.first_breaks_col, "PredictedFirstBreaks"], **kwargs)

    def get_views(self, sort_by=None, **kwargs):
        return [partial(self.plot_on_click, sort_by=sort_by)], kwargs


class MeanAbsoluteError(TravelTimeMetric):
    name = "mae"
    min_value = 0
    is_lower_better = True

    @staticmethod
    @njit(nogil=True)
    def calc(shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        _ = shots_coords, receivers_coords
        return np.abs(true_traveltimes - pred_traveltimes).mean()


class GeometryError(TravelTimeMetric):
    name = "geometry_error"
    min_value = 0
    is_lower_better = True

    @staticmethod
    def sin(x, amp, phase):
        return amp * np.sin(x + phase)

    @classmethod
    def calc(cls, shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        diff = true_traveltimes - pred_traveltimes
        x, y = (receivers_coords - shots_coords).T
        azimuth = np.arctan2(y, x)
        abs_diff = np.abs(diff)
        outlier_mask = abs_diff > np.quantile(abs_diff, 0.9)
        params = curve_fit(cls.sin, azimuth[~outlier_mask], diff[~outlier_mask], p0=[0, 0])[0]
        return abs(params[0])


TRAVELTIME_QC_METRICS = [MeanAbsoluteError, GeometryError]
