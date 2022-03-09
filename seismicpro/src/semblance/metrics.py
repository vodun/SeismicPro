import numpy as np

from ..metrics import PipelineMetric, pass_calc_args


class SignalLeakage(PipelineMetric):
    """Calculate signal leakage during ground-roll attenuation.

    The metric is based on the assumption that a vertical velocity semblance calculated for the difference between
    raw and processed gathers should not have pronounced energy maxima.

    Parameters
    ----------
    self : Semblance
        Semblance calculated for gather difference.
    other : Semblance
        Semblance for raw gather.

    Returns
    -------
    metric : float
        Signal leakage during gather processing.
    """
    name = "signal_leakage"
    min_value = 0
    max_value = None
    is_lower_better = True
    views = ("plot_diff_gather", "plot_diff_semblance")
    args_to_unpack = ("gather_before", "gather_after")

    @staticmethod
    def calc(gather_before, gather_after, velocities):
        gather_diff = gather_before.copy(ignore=["data", "headers", "samples"])
        gather_diff.data = gather_after.data - gather_before.data
        semblance_diff = gather_diff.calculate_semblance(velocities)
        semblance_before = gather_before.calculate_semblance(velocities)
        signal_leakage = semblance_diff.semblance.ptp(axis=1) / (1 + 1e-6 - semblance_before.semblance.ptp(axis=1))
        return max(0, np.max(signal_leakage))

    @pass_calc_args
    def plot_diff_gather(gather_before, gather_after, velocities, ax, **kwargs):
        _ = velocities
        gather_diff = gather_before.copy(ignore=["data", "headers", "samples"])
        gather_diff.data = gather_after.data - gather_before.data
        gather_diff.plot(ax=ax, **kwargs)

    @pass_calc_args
    def plot_diff_semblance(gather_before, gather_after, velocities, ax, **kwargs):
        gather_diff = gather_before.copy(ignore=["data", "headers", "samples"])
        gather_diff.data = gather_after.data - gather_before.data
        semblance_diff = gather_diff.calculate_semblance(velocities)
        semblance_diff.plot(ax=ax, **{"title": None, **kwargs})
