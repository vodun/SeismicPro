from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import KDTree
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
        self.knn = None

    def bind_context(self, metric_map, nsm, survey_list, first_breaks_col):
        self.nsm = nsm
        index_cols = metric_map.index_cols if len(survey_list) == 1 else metric_map.index_cols[1:]
        self.survey_list = [sur.reindex(index_cols) for sur in survey_list]
        self.first_breaks_col = first_breaks_col

        coords = pd.concat([sur.get_headers(metric_map.coords_cols) for sur in self.survey_list])        
        self.knn = KDTree(coords.drop_duplicates())

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
        gather = self.nsm.estimate_gather_traveltimes(gather)
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
                              bounds=((None, None), (None, None), (None, None)), tol=1e-2,
                              options=dict(initial_simplex=initial_simplex))
        
        return fit_result.x

    def correct_gather(self, gather):
        raise NotImplementedError

    def correct(self, metric_map, path=None):
        res = []
    
        if len(metric_map.index_cols) == 1:
            indices = pd.Index(metric_data[metric_map.index_cols])
        else:
            indices = pd.MultiIndex(metric_data[metric_map.index_cols])
        
        for index in tqdm(indices):
            gather = self.get_gather(index)
            corrected_gather = self.correct_gather(gather)
            before, after = np.array(gather.coords), np.array(corrected_gather.coords)
        
            if metric_map.index_cols != metric_map.coords_cols:
                cols = [*metric_map.index_cols, *metric_map.coords_cols, 'dx', 'dy', 'dxy']
                res.append([*to_list(index), *before, *(before - after), np.sqrt(np.sum((before - after) ** 2))])
            else:
                cols = [*metric_map.coords_cols, 'dx', 'dy', 'dxy']
                res.append([*before, *(before - after), np.sqrt(np.sum((before - after) ** 2))])
        
        info = pd.DataFrame(res, columns=cols).sort_values('dxy', ascending=False)
        if path is not None:
            info.to_string(path, float_format= lambda x: "{:.2f}".format(x), index=False)
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

    def plot_diff_by_azimuth(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        diff, azimuth = self.diff_by_azimuth(gather)
        
        params = self.fit(azimuth, diff)
        ax.scatter(azimuth, diff)
        ax.plot(np.linspace(-np.pi, np.pi, 100), self.sin(np.linspace(-np.pi, np.pi, 100), *params), c='k', label='new')
            
        val = max(np.quantile(np.abs(diff - params[-1]), 0.99), abs(params[0])) * 1.2
        ax.set_ylim(params[-1] - val, params[-1] + val)

    def plot_diff_by_azimuth_corrected(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        diff, azimuth = self.diff_by_azimuth(gather)

        params = GeometryErrorMs.fit(azimuth, diff)
        ax.scatter(azimuth, diff, c='b', s=10, label='before')
        azimuth = np.linspace(-np.pi, np.pi, 100)
        ax.plot(azimuth, GeometryErrorMs.sin(azimuth, *params), c='b')
    
        val = max(np.quantile(np.abs(diff - params[-1]), 0.99), abs(params[0])) * 1.2
        ax.set_ylim(params[-1] - val, params[-1] + val)

        gather = self.correct_gather(gather)
        gather = self.nsm.estimate_gather_traveltimes(gather)
        diff, azimuth = self.diff_by_azimuth(gather)

        params = GeometryErrorMs.fit(azimuth, diff)
        ax.scatter(azimuth, diff, c='r', s=2, label='after')
        azimuth = np.linspace(-np.pi, np.pi, 100)
        ax.plot(azimuth, GeometryErrorMs.sin(azimuth, *params), c='r')
        ax.legend()
    
    def plot_geometry_correction(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        corrected_gather = self.correct_gather(gather)
        before, after = np.array(gather.coords), np.array(corrected_gather.coords)

        r = max(np.sqrt(((before - after) ** 2).sum()), 500)

        if gather.is_shot:
            prime_label, prime_color, prime_marker, prime_size = ('Shots', 'r', '*', 1)
            minor_label, minor_color, minor_marker, minor_size = ('Recs', 'b', 'v', 36)
            minor_cols = ['GroupX', 'GroupY']
        elif gather.is_receiver:
            prime_label, prime_color, prime_marker, prime_size = ('Recs', 'b', 'v', 1)
            minor_label, minor_color, minor_marker, minor_size = ('Shots', 'r', '*', 36)
            minor_cols = ['SourceX', 'SourceY']

        ax.scatter(*gather[minor_cols].T, label=minor_label, c=minor_color, marker=minor_marker, s=minor_size)
        indices = self.knn.query_ball_point(gather.coords, r)
        ax.scatter(*self.knn.data[indices].T, label=prime_label, c=prime_color, marker=prime_marker, s=prime_size)

        ax.scatter(*before, label='Original Gather Position', c=prime_color)
        ax.arrow(*before, *(after - before), length_includes_head=False, head_width=10, color='g')
        ax.set_title(f'Total move {np.sqrt(np.sum((after - before) ** 2))}')

        ax.set_xlim(gather.coords[0] - r, gather.coords[0] + r)
        ax.set_ylim(gather.coords[1] - r, gather.coords[1] + r)
        ax.legend(loc='upper right')

    def plot_gather_after_correction(self, ax, coords, index, sort_by=None, **kwargs):
        _ = coords
        gather = self.get_gather(index, sort_by)
        corrected_gather = self.correct_gather(gather)
        corrected_gather = self.nsm.estimate_gather_traveltimes(corrected_gather)
        gather['Corrected ' + self.first_breaks_col] = corrected_gather["Predicted " + self.first_breaks_col]
        gather.plot(ax=ax, event_headers=dict(headers=["Predicted " + self.first_breaks_col, "Corrected " + self.first_breaks_col], c=['b', 'r'], s=10))

    def get_views(self, sort_by=None, **kwargs):
        if not self.corrected_views:
            return [self.plot_diff_by_azimuth, partial(self.plot_on_click, sort_by=sort_by)], kwargs
        else:
            return [self.plot_diff_by_azimuth_corrected, self.plot_geometry_correction, partial(self.plot_gather_after_correction, sort_by=sort_by)], kwargs


class GeometryErrorMs(BaseGeometryError):
    name = "geometry_error_ms"
    corrected_views = False

    def estimate_correction_velocity(self, coords):
        rv = self.nsm.rvf_list[0](coords)
        return np.array(list(rv.params.values()))[rv.n_refractors:].mean() / 1000
    
    def correct_gather(self, gather):
        gather = gather.copy()
        gather = self.nsm.estimate_gather_traveltimes(gather)
        diff, azimuth = self.diff_by_azimuth(gather)
        params = self.fit(azimuth, diff)

        v = self.estimate_correction_velocity(gather.coords)
        dx = -v * abs(params[0]) * np.sin(params[1]) * np.sign(params[0])
        dy = -v * abs(params[0]) * np.cos(params[1]) * np.sign(params[0])

        if gather.is_shot:
            gather.coords += np.array([dx, dy])
        elif gather.is_receiver:
            gather.coords -= np.array([dx, dy])
        gather.elevation = self.nsm.surface_elevation_interpolator(gather.coords)
        return gather

    def _calc(self, shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        diff, azimuth = self._diff_by_azimuth(shots_coords, receivers_coords, true_traveltimes, pred_traveltimes)
        params = self.fit(azimuth, diff)
        return abs(params[0])

    def __call__(self, gather):
        gather = self.nsm.estimate_gather_traveltimes(gather)
        return self._calc(gather['SourceX', 'SourceY'], gather['GroupX', 'GroupY'],
                          gather[self.first_breaks_col], gather['Predicted ' + self.first_breaks_col])



class GeometryErrorMeters(BaseGeometryError):

    name = 'geometry_error_meters'
    vmax = 30
    corrected_views = True

    def mean_diff(self, params, gather):
        gather.coords = params
        gather.elevation = self.nsm.surface_elevation_interpolator(gather.coords)
        gather = self.nsm.estimate_gather_traveltimes(gather)
        return np.abs(gather[self.first_breaks_col] - gather['Predicted ' + self.first_breaks_col]).mean()
                    
    def correct_gather(self, gather):        
        raw_corrected = self.metric_ms.correct_gather(gather)
    
        r = 100
        x0 = raw_corrected.coords
        initial_simplex = [(x0[0] + r * np.cos(alpha), x0[1] + r * np.sin(alpha)) for alpha in [0, 2*np.pi/3, 4*np.pi/3]]
        
        minimize(self.mean_diff, x0=gather.coords, args=(raw_corrected, ), method='Nelder-Mead', 
                 options=dict(xatol=1e-1, fatol=1e-1, initial_simplex=initial_simplex))
        return raw_corrected

    def __call__(self, gather):
        return np.sqrt(((np.array(gather.coords) - self.correct_gather(gather).coords) ** 2).sum())
    
    def calculate_map(self, metric_map):
        self.bind_context(metric_map, metric_map.metric.nsm, metric_map.metric.survey_list, 
                          metric_map.metric.first_breaks_col)
        self.has_bound_context = True
        self.metric_ms = metric_map.metric

        if len(metric_map.index_cols) == 1:
            indices = pd.Index(metric_data[metric_map.index_cols])
        else:
            indices = pd.MultiIndex(metric_data[metric_map.index_cols])

        values = [self(self.get_gather(index)) for index in tqdm(indices)]

        return self.construct_map(metric_map.metric_data[metric_map.coords_cols], values, 
                                  index=indices, agg=metric_map.agg)


TRAVELTIME_QC_METRICS = [MeanAbsoluteError, GeometryErrorMs]