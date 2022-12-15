from functools import partial

import numpy as np
from numba import njit
from scipy.optimize import minimize

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
        shots_coords = gather[["SourceX", "SourceY", "SourceSurfaceElevation"]]
        if is_uphole:
            shots_coords[:, -1] -= gather["SourceDepth"]
        fb_pred = self.nsm.estimate_traveltimes(shots_coords, gather[["GroupX", "GroupY", "ReceiverGroupElevation"]])
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
    def loss(cls, params, x, y, reg=0.01):
        return np.abs(y - cls.sin(x, *params)).mean() + reg * params[0]**2

    @classmethod
    def fit(cls, azimuth, diff, reg=0.01):
        fit_result = minimize(cls.loss, x0=[0, 0], args=(azimuth, diff - diff.mean(), reg),
                              bounds=((None, None), (-np.pi, np.pi)), method="Nelder-Mead", tol=1e-5)
        return fit_result.x

    @classmethod
    def calc(cls, shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        diff = true_traveltimes - pred_traveltimes
        x, y = (receivers_coords - shots_coords).T
        azimuth = np.arctan2(y, x)
        params = cls.fit(azimuth, diff)
        return abs(params[0])

    def plot_diff_by_azimuth(self, coords, ax, **kwargs):
        survey = [survey for survey in self.survey_list if coords in survey.indices][0]
        gather = survey.get_gather(coords, copy_headers=True)
        is_uphole = self.nsm.is_uphole
        if is_uphole is None:
            loaded_headers = set(survey.headers.columns) | set(survey.headers.index.names)
            is_uphole = "SourceDepth" in loaded_headers
        shots_coords = gather[["SourceX", "SourceY", "SourceSurfaceElevation"]]
        if is_uphole:
            shots_coords[:, -1] -= gather["SourceDepth"]
        fb_pred = self.nsm.estimate_traveltimes(shots_coords, gather[["GroupX", "GroupY", "ReceiverGroupElevation"]])

        diff = gather[self.nsm.first_breaks_col] - fb_pred
        x, y = (gather[["GroupX", "GroupY"]] - gather[["SourceX", "SourceY"]]).T
        azimuth = np.arctan2(y, x)

        params = self.fit(azimuth, diff)
        ax.scatter(azimuth, diff)
        azimuth = np.linspace(-np.pi, np.pi, 100)
        ax.plot(azimuth, diff.mean() + self.sin(azimuth, *params))

    def get_views(self, sort_by=None, **kwargs):
        return [partial(self.plot_on_click, sort_by=sort_by), self.plot_diff_by_azimuth], kwargs


TRAVELTIME_QC_METRICS = [MeanAbsoluteError, GeometryError]
