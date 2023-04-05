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


class GeometryError(TravelTimeMetric):
    name = "geometry_error"
    min_value = 0
    is_lower_better = True
    show_corrections = False

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

    def __call__(self, shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        diff = true_traveltimes - pred_traveltimes
        x, y = (receivers_coords - shots_coords).T
        azimuth = np.arctan2(y, x)
        params, n = self.fit(azimuth, diff)
        return abs(params[0])

    def estimate_traveltimes(self, gather):
        uphole_correction_method = self.nsm._get_uphole_correction_method(self.survey_list[0])
        source_coords, receiver_coords, correction = self.nsm._get_predict_traveltime_data(gather, uphole_correction_method)
        pred_traveltimes = self.nsm.estimate_traveltimes(source_coords, receiver_coords) - correction
        gather["Predicted " + self.first_breaks_col] = pred_traveltimes
        return gather

    def correct_raw(self, gather):
        new_gather = gather.copy()
        new_gather = self.estimate_traveltimes(new_gather)
        diff = new_gather[self.first_breaks_col] - new_gather["Predicted " + self.first_breaks_col]
        x, y = (new_gather[["GroupX", "GroupY"]] - new_gather[["SourceX", "SourceY"]]).T
        azimuth = np.arctan2(y, x)
        params = self.fit(azimuth, diff)
        
        v = self.nsm.rvf_list[0](gather.coords).v1 / 1000
        dx = -v * abs(params[0]) * np.sin(params[1]) * np.sign(params[0])
        dy = -v * abs(params[0]) * np.cos(params[1]) * np.sign(params[0])
        
        new_gather[new_gather.coords.names] += np.array([dx, dy])

        if new_gather.coords.names == ('SourceX', 'SourceY'):
            new_gather['SourceSurfaceElevation'] = self.nsm.surface_elevation_interpolator(new_gather.coords)
        elif gather.coords.names == ('GroupX', 'GroupY'):
            new_gather['ReceiverGroupElevation'] = self.nsm.surface_elevation_interpolator(gnew_gatherther.coords)
        return new_gather
    
    def correct_gather(self, gather):
        gather = gather.copy()
        
        d = 100
        x0 = gather.coords
        initial_simplex = [(x0[0] + d * np.cos(alpha), x0[1] + d * np.sin(alpha)) for alpha in [0, 2*np.pi/3, 4*np.pi/3]]
        
        fit_result = minimize(self.correction_loss, x0=gather.coords, args=(gather, ), method='Nelder-Mead', 
                              options=dict(xatol=5, fatol=3, initial_simplex=initial_simplex))
        gather[list(gather.coords.names)] = fit_result.x
        return gather
    
    def correction_loss(self, params, gather):
        gather[list(gather.coords.names)] = params

        if gather.coords.names == ('SourceX', 'SourceY'):
            gather['SourceSurfaceElevation'] = self.nsm.surface_elevation_interpolator(gather.coords)
        elif gather.coords.names == ('GroupX', 'GroupY'):
            gather['ReceiverGroupElevation'] = self.nsm.surface_elevation_interpolator(gather.coords)

        gather = self.estimate_traveltimes(gather)
        return np.abs(gather[self.first_breaks_col] - gather['Predicted ' + self.first_breaks_col]).mean()

    def correct(self, metric_map, path=None):
        info = []
        for _, row in tqdm(metric_map.index_data.iterrows()):
            row = row[[*metric_map.index_cols, metric_map.metric_name]].values
            index, amp = row[:-1].astype(np.int32), row[-1]

            if len(metric_map.index_cols) == 1:
                index = int(index)
            else:
                index = tuple(index)
    
            gather = self.get_gather(index)
            gather = self.correct_raw(gather)
            corrected_gather = self.correct_gather(gather)

            before, after = np.array(gather.coords), np.array(corrected_gather.coords)

            if gather.coords.names == ('SourceX', 'SourceY'):
                title = 'Shot Geometry'
                cols = [*metric_map.index_cols, 'x', 'y', 'dx', 'dy', 'amp', 'dxy']
                info.append([*to_list(index), *before, *(before - after), amp, np.sqrt(np.sum((before - after) ** 2))])
            elif gather.coords.names == ('GroupX', 'GroupY'):
                title = 'Rec geometry'
                cols = ['x', 'y', 'dx', 'dy', 'amp', 'dxy']
                info.append([*before, *(before - after), amp, np.sqrt(np.sum((before - after) ** 2))])

        return pd.DataFrame(info, columns=cols).sort_values('dxy', ascending=False).to_string(path, float_format= lambda x: "{:.2f}".format(x), index=False)


    def plot_diff_by_azimuth(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        diff = gather[self.first_breaks_col] - gather["Predicted " + self.first_breaks_col]
        x, y = (gather[["GroupX", "GroupY"]] - gather[["SourceX", "SourceY"]]).T
        azimuth = np.arctan2(y, x)
        
        params, q = self.fit(azimuth, diff)
        ax.scatter(azimuth, diff)
        ax.plot(np.linspace(-np.pi, np.pi, 100), self.sin(np.linspace(-np.pi, np.pi, 100), *params), c='k', label='new')
        
        params, p = MasterGeometryError.fit(azimuth, diff, 0)
        ax.plot(np.linspace(-np.pi, np.pi, 100), MasterGeometryError.sin(np.linspace(-np.pi, np.pi, 100), *params) + diff.mean(), c='r', label='before')
        ax.set_title(f'before {p}, new {q}')
        ax.legend()
    
        val = max(np.quantile(np.abs(diff - params[-1]), 0.99), abs(params[0])) * 1.2
        ax.set_ylim(params[-1] - val, params[-1] + val)

    def plot_diff_by_azimuth_corrected(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        diff = gather[self.first_breaks_col] - gather["Predicted " + self.first_breaks_col]
        x, y = (gather[["GroupX", "GroupY"]] - gather[["SourceX", "SourceY"]]).T
        azimuth = np.arctan2(y, x)

        params, n = self.fit(azimuth, diff)
        ax.scatter(azimuth, diff, c='b', s=10, label='before')
        azimuth = np.linspace(-np.pi, np.pi, 100)
        ax.plot(azimuth, self.sin(azimuth, *params), c='b')
    
        val = max(np.quantile(np.abs(diff - params[-1]), 0.99), abs(params[0])) * 1.2
        ax.set_ylim(params[-1] - val, params[-1] + val)

        gather = self.correct_gather(gather)
        diff = gather[self.first_breaks_col] - gather["Predicted " + self.first_breaks_col]
        x, y = (gather[["GroupX", "GroupY"]] - gather[["SourceX", "SourceY"]]).T
        azimuth = np.arctan2(y, x)

        params, n = self.fit(azimuth, diff)
        ax.scatter(azimuth, diff, c='r', s=2, label='after')
        azimuth = np.linspace(-np.pi, np.pi, 100)
        ax.plot(azimuth, self.sin(azimuth, *params), c='r')
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
        if not self.show_corrections:
            return [self.plot_diff_by_azimuth, partial(self.plot_on_click, sort_by=sort_by)], kwargs
        else:
            return [self.plot_diff_by_azimuth_corrected, self.plot_geometry_correction, partial(self.plot_gather_after_correction, sort_by=sort_by)], kwargs


class MasterGeometryError(TravelTimeMetric):
    name = "master_geometry_error"
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
        return fit_result.x, fit_result.nit

    def __call__(self, shots_coords, receivers_coords, true_traveltimes, pred_traveltimes):
        diff = true_traveltimes - pred_traveltimes
        x, y = (receivers_coords - shots_coords).T
        azimuth = np.arctan2(y, x)
        params, n = self.fit(azimuth, diff, reg=self.reg)
        return abs(params[0])

    def plot_diff_by_azimuth(self, ax, coords, index, **kwargs):
        _ = coords
        gather = self.get_gather(index)
        diff = gather[self.first_breaks_col] - gather["Predicted " + self.first_breaks_col]
        x, y = (gather[["GroupX", "GroupY"]] - gather[["SourceX", "SourceY"]]).T
        azimuth = np.arctan2(y, x)

        params, p = self.fit(azimuth, diff, 0)
        ax.scatter(azimuth, diff)
        ax.plot(np.linspace(-np.pi, np.pi, 100), diff.mean() + self.sin(np.linspace(-np.pi, np.pi, 100), *params), c='r', label='before')
        
        params, q = GeometryError.fit(azimuth, diff)
        ax.plot(np.linspace(-np.pi, np.pi, 100), GeometryError.sin(np.linspace(-np.pi, np.pi, 100), *params), c='k', label='new')
        ax.legend()
        ax.set_title(f'before {int(p)}, new {int(q)}')

    def get_views(self, sort_by=None, **kwargs):
        return [partial(self.plot_on_click, sort_by=sort_by), self.plot_diff_by_azimuth], kwargs

TRAVELTIME_QC_METRICS = [MeanAbsoluteError, GeometryError]
