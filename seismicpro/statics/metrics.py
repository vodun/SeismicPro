from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize

from ..metrics import Metric
from ..utils import to_list


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

    def estimate_traveltimes(self, gather):
        uphole_correction_method = self.nsm._get_uphole_correction_method(self.survey_list[0])
        source_coords, receiver_coords, correction = self.nsm._get_predict_traveltime_data(gather, uphole_correction_method)
        pred_traveltimes = self.nsm.estimate_traveltimes(source_coords, receiver_coords) - correction
        gather["Predicted " + self.first_breaks_col] = pred_traveltimes
        return gather

    def get_gather(self, index, sort_by=None):
        if len(self.survey_list) == 1:
            part = 0
        else:
            part = index[0]
            index = index[1:]
        survey = self.survey_list[part]
        gather = survey.get_gather(index, copy_headers=True)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        uphole_correction_method = self.nsm._get_uphole_correction_method(survey)
        source_coords, receiver_coords, correction = self.nsm._get_predict_traveltime_data(gather, uphole_correction_method)
        pred_traveltimes = self.nsm.estimate_traveltimes(source_coords, receiver_coords, bar=False) - correction
        gather["Predicted " + self.first_breaks_col] = pred_traveltimes
        return gather

    def plot_on_click(self, ax, coords, index, sort_by=None, **kwargs):
        _ = coords
        gather = self.get_gather(index, sort_by)
        gather.plot(ax=ax, event_headers=[self.nsm.first_breaks_col, "Predicted " + self.first_breaks_col], **kwargs)

    def get_views(self, sort_by=None, **kwargs):
        return [partial(self.plot_on_click, sort_by=sort_by)], kwargs



class MeanAbsoluteError(TravelTimeMetric):
    name = "mae"
    min_value = 0
    is_lower_better = True

    def __call__(self, shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        _ = shots_coords, receivers_coords
        return np.abs(true_traveltimes - pred_traveltimes).mean()


class BaseGeometryError(TravelTimeMetric):
    min_value = 0
    is_lower_better = True

    def correct_gather(self, gather):
        raise NotImplementedError

    def correct(self, metric_map, path=None):
        res = []
        indices = metric_map.metric_data.set_index(metric_map.index_cols).index
        for index in tqdm(indices):
            gather = self.get_gather(index)
            corrected_gather = self.correct_gather(gather)
            before, after = np.array(gather.coords), np.array(corrected_gather.coords)
            cols = [*metric_map.index_cols, *metric_map.coords_cols, 'dx', 'dy', 'dxy']
            res.append([*to_list(index), *before, *(before - after), np.sqrt(np.sum((before - after) ** 2))])
        info = pd.DataFrame(res, columns=cols).sort_values('dxy', ascending=False).to_string(path, float_format= lambda x: "{:.2f}".format(x), index=False)
        if path is not None:
            with open(path, 'w') as f:
                f.write(df)
        return info

    def diff_by_azimuth(self, gather):
        return self._diff_by_azimuth(gather['SourceX', 'SourceY'], gather['GroupX', 'GroupY'],
                                     gather[self.first_breaks_col], gather['Predicted ' + self.first_breaks_col])

    @staticmethod
    def _diff_by_azimuth(shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        diff = true_traveltimes - pred_traveltimes
        x, y = (receivers_coords - shots_coords).T
        azimuth = np.arctan2(y, x)
        return diff, azimuth


class GeometryErrorMs(BaseGeometryError):
    name = "geometry_error_ms"

    @staticmethod
    def sin(x, amp, phase, c):
        return amp * np.sin(x + phase) + c

    @classmethod
    def loss(cls, params, x, y, reg):
        return np.abs(y - cls.sin(x, *params)).mean() + reg * np.abs(params[0])

    @classmethod
    def fit(cls, azimuth, diff):
        N = 18
        a, b = np.histogram(azimuth, bins=np.linspace(-np.pi, np.pi, N + 1))
        mean = len(azimuth) / N
        reg = np.mean(a < mean * 0.2) * 0.25
                
        damp = 5
        dphase = np.pi
        dt = 5
        initial_simplex = np.array([
                        [damp, dphase, diff.mean() + dt], 
                        [damp, -dphase / 2, diff.mean() + dt], 
                        [-damp, dphase / 2, diff.mean() + dt], 
                        [-damp, 0, diff.mean() -dt], 
                    ])
        fit_result = minimize(cls.loss, x0=[0, 0, diff.mean()], args=(azimuth, diff, reg), method="Nelder-Mead", 
                              bounds=((None, None), (-np.pi, np.pi), (None, None)), tol=1e-2,
                              options=dict(initial_simplex=initial_simplex))
        
        return fit_result.x, fit_result.nit


    def _calc(self, shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        diff, azimuth = self._diff_by_azimuth(shots_coords, receivers_coords, true_traveltimes, pred_traveltimes)
        params, n = self.fit(azimuth, diff)
        return abs(params[0])

    def __call__(self, gather):
        gather = self.estimate_traveltimes(gather)
        return self._calc(gather['SourceX', 'SourceY'], gather['GroupX', 'GroupY'],
                          gather[self.first_breaks_col], gather['Predicted ' + self.first_breaks_col])

    def correct_gather(self, gather):
        gather = gather.copy()
        gather = self.estimate_traveltimes(gather)
        diff, azimuth = self.diff_by_azimuth(gather)
        params, nit = self.fit(azimuth, diff)
        
        v = self.nsm.rvf_list[0](gather.coords).v1 / 1000
        dx = -v * params[0] * np.sin(params[1])
        dy = -v * params[0] * np.cos(params[1])
        
        gather.coords += np.array([dx, dy])
        gather.elevation = self.nsm.surface_elevation_interpolator(gather.coords)
        return gather

    def plot_diff_by_azimuth(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        diff, azimuth = self.diff_by_azimuth(gather)
        
        params, q = self.fit(azimuth, diff)
        ax.scatter(azimuth, diff)
        ax.plot(np.linspace(-np.pi, np.pi, 100), self.sin(np.linspace(-np.pi, np.pi, 100), *params), c='k', label='new')
            
        val = max(np.quantile(np.abs(diff - params[-1]), 0.99), abs(params[0])) * 1.2
        ax.set_ylim(params[-1] - val, params[-1] + val)

    def get_views(self, sort_by=None, **kwargs):
        return [self.plot_diff_by_azimuth, partial(self.plot_on_click, sort_by=sort_by)], kwargs


class GeometryErrorMeters(BaseGeometryError):

    name = 'geometry_error_meters'
    vmax = 30

    def loss(self, params, gather):
        gather.coords = params
        gather.elevation = self.nsm.surface_elevation_interpolator(gather.coords)
        gather = self.estimate_traveltimes(gather)
        return np.abs(gather[self.first_breaks_col] - gather['Predicted ' + self.first_breaks_col]).mean()
                    
    def correct_gather(self, gather):
        gather = gather.copy()
        
        d = 100
        x0 = gather.coords
        initial_simplex = [(x0[0] + d * np.cos(alpha), x0[1] + d * np.sin(alpha)) for alpha in [0, 2*np.pi/3, 4*np.pi/3]]
        
        minimize(self.loss, x0=gather.coords, args=(gather, ), method='Nelder-Mead', 
                 options=dict(xatol=5, fatol=self.nsm.loss, initial_simplex=initial_simplex))
        return gather

    def __call__(self, gather):
        return np.sqrt(((np.array(gather.coords) - self.correct_gather(gather).coords) ** 2).sum())
    
    def calculate_map(self, metric_map):

        self.nsm = metric_map.metric.nsm
        self.survey_list = metric_map.metric.survey_list
        self.first_breaks_col = metric_map.metric.first_breaks_col
        self.has_bound_context = True

        indices = metric_map.metric_data.set_index(metric_map.index_cols).index
        values = [self(self.get_gather(index)) for index in tqdm(indices)]

        return self.construct_map(metric_map.metric_data[metric_map.coords_cols], values, 
                                  index=indices, agg=metric_map.agg)

    def plot_diff_by_azimuth_corrected(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        diff, azimuth = self.diff_by_azimuth(gather)

        params, n = GeometryErrorMs.fit(azimuth, diff)
        ax.scatter(azimuth, diff, c='b', s=10, label='before')
        azimuth = np.linspace(-np.pi, np.pi, 100)
        ax.plot(azimuth, GeometryErrorMs.sin(azimuth, *params), c='b')
    
        val = max(np.quantile(np.abs(diff - params[-1]), 0.99), abs(params[0])) * 1.2
        ax.set_ylim(params[-1] - val, params[-1] + val)

        gather = self.correct_gather(gather)
        diff, azimuth = self.diff_by_azimuth(gather)

        params, n = GeometryErrorMs.fit(azimuth, diff)
        ax.scatter(azimuth, diff, c='r', s=2, label='after')
        azimuth = np.linspace(-np.pi, np.pi, 100)
        ax.plot(azimuth, GeometryErrorMs.sin(azimuth, *params), c='r')
        ax.legend()
    
    def plot_geometry_correction(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        corrected_gather = self.correct_gather(gather)
        before, after = np.array(gather.coords), np.array(corrected_gather.coords)

        ax.scatter(*gather['GroupX', 'GroupY'].T, label='Recs', c='b', marker='*')

        xy = pd.concat([survey.get_headers(gather.coords.names) for survey in self.survey_list])
        xy = xy.drop_duplicates().values
        from scipy.spatial import KDTree
        tree = KDTree(xy)
        d = 500
        indices = tree.query_ball_point(gather.coords, 500)
        ax.scatter(*xy[indices].T, label='Shots', c='r', s=1)


        ax.scatter(*before, label='Original Shot Position', c='r')
        ax.arrow(*before, *(after - before), length_includes_head=False, head_width=10, color='g')
        ax.set_title(f'Total move {np.sqrt(np.sum((after - before) ** 2))}')

        ax.set_xlim(gather.coords[0] - d, gather.coords[0] + d)
        ax.set_ylim(gather.coords[1] - d, gather.coords[1] + d)
        ax.legend(loc='upper right')

    def plot_gather_after_correction(self, ax, coords, index, sort_by=None, **kwargs):
        _ = coords
        gather = self.get_gather(index, sort_by)
        corrected_gather = self.correct_gather(gather)
        corrected_gather = self.estimate_traveltimes(corrected_gather)
        gather['Corrected ' + self.first_breaks_col] = corrected_gather["Predicted " + self.first_breaks_col]
        gather.plot(ax=ax, event_headers=dict(headers=["Predicted " + self.first_breaks_col, "Corrected " + self.first_breaks_col], c=['b', 'r'], s=10))

    def get_views(self, sort_by=None, **kwargs):
        return [self.plot_diff_by_azimuth_corrected, self.plot_geometry_correction, partial(self.plot_gather_after_correction, sort_by=sort_by)], kwargs


TRAVELTIME_QC_METRICS = [MeanAbsoluteError, GeometryErrorMs]