from functools import partial

import numpy as np
from numba import njit
from scipy.optimize import minimize

from ..metrics import Metric


class TravelTimeMetric(Metric):
    def __init__(self, name=None):
        super().__init__(name=name)

        # Attributes set after context binding
        self.nsm = None
        self.survey_list = None
        self.first_breaks_col = None

    def bind_context(self, metric_map, nsm, survey_list, first_breaks_col):
        self.nsm = nsm
        index_cols = metric_map.index_cols if len(survey_list) == 1 else metric_map.index_cols[1:]
        self.survey_list = [sur.reindex(index_cols) for sur in survey_list]
        self.first_breaks_col = first_breaks_col

    def get_gather(self, index, sort_by=None):
        part, index = (0, index) if len(self.survey_list) == 1 else index
        gather = self.survey_list[part].get_gather(index, copy_headers=True)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        uphole_correction_method = self.nsm._get_uphole_correction_method(gather.sur)
        source_coords, receiver_coords, correction = self.nsm._get_predict_traveltime_data(gather, uphole_correction_method)
        pred_traveltimes = self.estimate_traveltimes(source_coords, receiver_coords, bar=False) - correction
        gather["Predicted " + self.first_breaks_col] = pred_traveltimes
        # Shift by uphole if needed
        return gather

    def plot_on_click(self, ax, coords, index, sort_by=None, **kwargs):
        _ = coords
        gather = self.get_gather(index, sort_by)
        gather.plot(ax=ax, event_headers=[self.nsm.first_breaks_col, "PredictedFirstBreaks"], **kwargs)

    def get_views(self, sort_by=None, **kwargs):
        return [partial(self.plot_on_click, sort_by=sort_by)], kwargs


class MeanAbsoluteError(TravelTimeMetric):
    name = "mae"
    min_value = 0
    is_lower_better = True

    def __call__(self, shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        _ = shots_coords, receivers_coords
        return np.abs(true_traveltimes - pred_traveltimes).mean()


class GeometryError(TravelTimeMetric):
    name = "geometry_error"
    min_value = 0
    is_lower_better = True

    def __init__(self, reg=0.01, name=None):
        self.reg = reg
        super().__init__(name=name)

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(reg={self.reg}, name='{self.name}')"

    @staticmethod
    def sin(x, amp, phase):
        return amp * np.sin(x + phase)

    @classmethod
    def loss(cls, params, x, y, reg):
        return np.abs(y - cls.sin(x, *params)).mean() + reg * params[0]**2

    @classmethod
    def fit(cls, azimuth, diff, reg):
        fit_result = minimize(cls.loss, x0=[0, 0], args=(azimuth, diff - diff.mean(), reg),
                              bounds=((None, None), (-np.pi, np.pi)), method="Nelder-Mead", tol=1e-5)
        return fit_result.x

    def __call__(self, shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        diff = true_traveltimes - pred_traveltimes
        x, y = (receivers_coords - shots_coords).T
        azimuth = np.arctan2(y, x)
        params = self.fit(azimuth, diff, reg=self.reg)
        return abs(params[0])

    def plot_diff_by_azimuth(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        diff = gather[self.first_breaks_col] - gather["Predicted " + self.first_breaks_col]
        x, y = (gather[["GroupX", "GroupY"]] - gather[["SourceX", "SourceY"]]).T
        azimuth = np.arctan2(y, x)

        params = self.fit(azimuth, diff)
        ax.scatter(azimuth, diff)
        azimuth = np.linspace(-np.pi, np.pi, 100)
        ax.plot(azimuth, diff.mean() + self.sin(azimuth, *params))

    def get_views(self, sort_by=None, **kwargs):
        return [partial(self.plot_on_click, sort_by=sort_by), self.plot_diff_by_azimuth], kwargs


TRAVELTIME_QC_METRICS = [MeanAbsoluteError, GeometryError]
