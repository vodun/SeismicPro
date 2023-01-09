import numpy as np
from numba import njit
from matplotlib import patches
from scipy.signal import hilbert
import torch

from ..metrics import Metric, ScatterMapPlot, MetricMap
from ..const import HDR_FIRST_BREAK


class RefractorVelocityMetric(Metric):
    views = ("plot_gather", "plot_refractor_velocity")

    def __init__(self, survey=None, field=None,
                    first_breaks_col=HDR_FIRST_BREAK, threshold_times=50, **kwargs):
        super().__init__(**kwargs)
        self.survey = survey
        self.field = field
        self.first_breaks_col = first_breaks_col
        self.threshold_times = threshold_times

    def plot_gather(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        event_headers = kwargs.pop('event_headers', {'headers': self.first_breaks_col})
        gather.plot(event_headers=event_headers, ax=ax, **kwargs)
        
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        refractor_velocity = self.field(coords)
        if not refractor_velocity.is_fit:
            gather_headers = self.survey.get_gather(coords)
            refractor_velocity.times = gather_headers[self.first_breaks_col]
            refractor_velocity.offsets = gather_headers['offset']
        refractor_velocity.plot(threshold_times=self.threshold_times, ax=ax, **kwargs)

        
class FirstBreaksOutliers(RefractorVelocityMetric):
    name = "first_breaks_outliers"
    vmin = 0
    vmax = 0.05
    is_lower_better = True
    
    @staticmethod
    def _calc(times, offsets, rv, threshold_times):
        return np.mean(np.abs(rv(offsets) - times) > threshold_times)
    
    @classmethod
    def calc(cls, gathers, refractor_velocities, threshold_times=50, first_breaks_col=HDR_FIRST_BREAK):
        metric = []
        for gather, rv in zip(gathers, refractor_velocities):
            metric.append(cls._calc(gather[first_breaks_col], gather['offset'], rv, threshold_times))
        return metric 


class FirstBreaksAmplitudes(RefractorVelocityMetric):
    name = "first_breaks_amplitudes"
    vmin = 0
    vmax = 3
    is_lower_better = True # ?
    
    @classmethod
    def calc(cls, gathers, refractor_velocities, first_breaks_col=HDR_FIRST_BREAK):
        _ = refractor_velocities
        metric = []
        for gather in gathers:
            ix = np.round(gather[first_breaks_col] / gather.sample_rate).astype(int)
            gather['amps'] = gather.data[np.arange(gather.shape[0]), ix]
            gather.store_headers_to_survey(columns='amps')
            metric.append(gather['amps'].mean())
        return metric
    
    def plot_gather(self, coords, ax, **kwargs):
        kwargs['top_header'] = 'amps'
        super().plot_gather(coords, ax, **kwargs)


class FirstBreaksPhases(RefractorVelocityMetric):
    name = 'first_breaks_phases'
    vmin = -np.pi / 2
    vmax = np.pi / 2
    is_lower_better = False
    
    @classmethod
    def calc(cls, gathers, refractor_velocities, first_breaks_col=HDR_FIRST_BREAK):
        _ = refractor_velocities
        metric = []
        for gather in gathers:
            ix = np.round(gather[first_breaks_col] / gather.sample_rate).astype(int)
            phases = hilbert(gather.data, axis=1)
            angles = np.angle(phases)
            gather.headers['phase'] = angles[np.arange(gather.shape[0]), ix]
            gather.store_headers_to_survey(columns='phase')
            metric.append(gather['phase'].mean())
        return metric

    def plot_gather(self, coords, ax, **kwargs):
        kwargs['top_header'] = 'phase'
        super().plot_gather(coords, ax, **kwargs)


class FirstBreaksCorrelations(RefractorVelocityMetric):
    name = "first_breaks_correlations"
    vmin = 0.5
    vmax = 1
    is_lower_better = False
    
    @classmethod
    def calc(cls, gathers, refractor_velocities, first_breaks_col=HDR_FIRST_BREAK, corr_window=10):
        _ = refractor_velocities
        metric = []
        for gather in gathers:
            ix = np.round(gather[first_breaks_col].ravel() / gather.sample_rate).astype(int)
            corrs = []
            prev = gather.data[0, ix[0] - corr_window: ix[0] + corr_window]
            for trace, pick in list(zip(gather.data, ix))[1:]:
                current = trace[pick - corr_window: pick + corr_window]
                current = np.pad(current, (max(corr_window - pick, 0), max(pick + corr_window - gather.shape[1], 0)))
                coef = np.corrcoef(current, prev)[0, 1]
                prev = current 
                corrs.append(abs(coef))
            corrs.append(1)
            gather['corr'] = corrs
            gather.store_headers_to_survey(columns='corr')
            metric.append(gather['corr'].mean())
        return metric
    
    def plot_gather(self, coords, ax, **kwargs):
        kwargs['top_header'] = 'corr'
        super().plot_gather(coords, ax, **kwargs)


class AzimuthDependency(RefractorVelocityMetric):
    name = "azimuth_dependency"
    vmin = 0
    vmax = 10
    is_lower_better = True
    
    @staticmethod
    def _fit(azimuth, diff, reg_coef, max_iter, n_bad_iters):
        azimuth = torch.tensor(azimuth, dtype=torch.float32)
        diff = torch.tensor(diff, dtype=torch.float32)
        weights = torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([weights], lr=0.25)

        best_loss = None
        best_weights = None
        bad_counter = 0

        for i in range(max_iter):
            optimizer.zero_grad()
            pred = weights[0] + weights[1] * torch.sin(azimuth + weights[2])
            loss = (pred - diff).abs().mean() + reg_coef * weights[1]**2
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            if best_loss is None or loss_item < best_loss:
                bad_counter = 0
                best_loss = loss_item
                best_weights = weights.detach().numpy()
            else:
                bad_counter += 1
                if bad_counter >= n_bad_iters:
                    break
        return best_weights, best_loss
    
    @classmethod
    def calc(cls, gathers, refractor_velocities, first_breaks_col=HDR_FIRST_BREAK,
             reg_coef=0.001, max_iter=250, n_bad_iters=10):
        metric = []
        for gather, rv in zip(gathers, refractor_velocities):
            vector = gather['GroupX', 'GroupY'] - gather['SourceX', 'SourceY']
            vector = vector[:, 0] + 1j * vector[:, 1]

            angle = np.angle(vector)
            diff = gather[first_breaks_col] - rv(gather.offsets)
            weights, loss = cls._fit(angle, diff, reg_coef=reg_coef, max_iter=max_iter, n_bad_iters=n_bad_iters)

            metric.append(loss)
            gather['azimuth'] = angle
            gather['azimuth_weights'] = [weights] * len(gather.data)
            gather['azimuth_loss'] = loss
            gather.store_headers_to_survey(columns=['azimuth', 'azimuth_weights', 'azimuth_loss'])
        return metric

    def plot_refractor_velocity(self, coords, ax, **kwargs):
        refractor_velocity = self.field(coords)
        gather = self.survey.get_gather(coords)
        diff = gather[self.first_breaks_col] - refractor_velocity(gather.offsets)  
        ax.scatter(gather['azimuth'], diff)

        x = np.linspace(-np.pi, np.pi, 100)
        weights = gather['azimuth_weights'][0]
        ax.plot(x, weights[0] + weights[1] * np.sin(x + weights[2]), c='r')


class DivergencePoint(RefractorVelocityMetric):
    """Find offset after that first breaks are most likely to significantly diverge from expected time.
    Such an offset is found as one with the maximum increase of outliers after it(actually, maximum increase of 
    increase of outliers, as it seems to be more robust).
    Divergence point is set to be the maximum offset when the overall fraction of outliers does not exceeds given tolerance.
    """
    name = "divergence_point"
    is_lower_better = False
    
    def __init__(self, survey=None, **kwargs):
        super().__init__(survey, **kwargs)
        self.vmax = survey['offset'].max() if survey is not None else None
        self.vmin = survey['offset'].min() if survey is not None else None

    @staticmethod
    def _calc(times, offsets, rv, threshold_times, offset_step):
        sorted_offsets_idx = np.argsort(offsets)
        offsets = offsets[sorted_offsets_idx]
        times = times[sorted_offsets_idx]
        
        check_offsets = list(range(offsets.min() + offset_step, max(offsets) - offset_step, offset_step))
        check_offsets += list(range(offsets.min() + int(offset_step * 0.25),
                                    offsets.max() - offset_step, offset_step))
        check_offsets += list(range(offsets.min() + int(offset_step * 0.5),
                                    offsets.max() - offset_step, offset_step))
        
        check_offsets.sort()
        check_indices = np.searchsorted(offsets, check_offsets)
        n = len(check_indices)
        
        time_windows = [times[:idx] for idx in check_indices] + [times[idx:] for idx in check_indices] 
        offset_windows = [offsets[:idx] for idx in check_indices] + [offsets[idx:] for idx in check_indices]
        outliers = [FirstBreaksOutliers._calc(time_window, offset_window, rv, threshold_times=threshold_times) 
                    for time_window, offset_window in zip(time_windows, offset_windows)]
        
        diffs = [outliers[i + n] - outliers[i] for i in range(n)]
        diffs_delta = [diffs[i + 1] - diffs[i] for i in range(n - 1)]
        return check_offsets[np.argmax(diffs_delta) + 1]
    
    @classmethod
    def calc(cls, gathers, refractor_velocities, first_breaks_col=HDR_FIRST_BREAK, # divide by num points, not offset?
             threshold_times=50, max_offset=None, tol=0.03, offset_step=100):
        outliers = FirstBreaksOutliers.calc(gathers, refractor_velocities, first_breaks_col=first_breaks_col, 
                                            threshold_times=threshold_times)
        
        max_offset = np.max(np.concatenate([gather['offset'] for gather in gathers])) if max_offset is None else max_offset
        metric = np.where(np.array(outliers) <= tol, max_offset, np.nan)
        tol_idx = np.nonzero(np.array(outliers) > tol)[0]
        for idx in tol_idx:
            metric[idx] = cls._calc(times=gathers[idx][first_breaks_col], offsets=gathers[idx]['offset'],
                                    rv=refractor_velocities[idx],
                                    threshold_times=threshold_times, offset_step=offset_step)
        for gather, divergence_offset in zip(gathers, metric):
            gather['divergence'] = divergence_offset
        gather.store_headers_to_survey(columns='divergence')
        return metric
        
    def plot_refractor_velocity(self, coords, ax, **kwargs):
        x_coord = self.survey.get_gather(coords)['divergence'][0]
        ax.axvline(x=x_coord, color='k', linestyle='--')
        super().plot_refractor_velocity(coords, ax, **kwargs)
        
REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, DivergencePoint, FirstBreaksAmplitudes, FirstBreaksPhases,
                                 FirstBreaksCorrelations, AzimuthDependency]
