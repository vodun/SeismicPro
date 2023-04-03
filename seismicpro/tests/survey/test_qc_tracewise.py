"""Test qc_tracewise and construct_qc_map methods"""

import pytest
import numpy as np

from seismicpro import Survey, Gather, Muter
from seismicpro.refractor_velocity import RefractorVelocity
from seismicpro.survey.metrics import (DeadTrace, TraceAbsMean, TraceMaxAbs, MaxClipsLen, MaxConstLen, Spikes,
                                       Autocorrelation, WindowRMS, AdaptiveWindowRMS, DEFAULT_TRACEWISE_METRICS)
from seismicpro.utils import to_list
from . import assert_surveys_equal, assert_survey_processed_inplace


rv = RefractorVelocity(t0=0, v1=1600)
muter = Muter.from_refractor_velocity(rv)


METRICS_TO_COMPUTE = [
    [None, [default_metric.name for default_metric in DEFAULT_TRACEWISE_METRICS]],
    [MaxClipsLen, MaxClipsLen.name],
    [[TraceAbsMean], TraceAbsMean.name],
    [[TraceMaxAbs()], TraceMaxAbs.name],
    [[TraceMaxAbs(name='first_maxabs'), TraceMaxAbs(name='second_maxabs')], ['first_maxabs', 'second_maxabs']],
    [
        [DeadTrace(), MaxClipsLen(), MaxConstLen(), Spikes(muter=muter), Autocorrelation(muter=muter)],
        [DeadTrace.name, MaxClipsLen.name, MaxConstLen.name, Spikes.name, Autocorrelation.name]
    ],
    [
        [WindowRMS(offsets=[0, 100], times=[0, 100]), AdaptiveWindowRMS(20, shift=100, refractor_velocity=rv)],
        [WindowRMS.name, AdaptiveWindowRMS.name]
    ]
]


MMAP_METRICS = [
    [None, None],
    [MaxClipsLen, MaxClipsLen.name],
    [[DeadTrace(), MaxClipsLen()], [DeadTrace.name, MaxClipsLen.name]],
    [
        [WindowRMS(offsets=[0, 100], times=[300, 500], name="signal_rms"),
         WindowRMS(offsets=[0, 100], times=[0, 100], name="noise_rms")],
        ["signal_rms", "noise_rms", "signal_rms/noise_rms"]
    ],
    [
        [AdaptiveWindowRMS(window_size=20, shift=100, refractor_velocity=rv, name="signal_rms"),
         AdaptiveWindowRMS(window_size=20, shift=-100, refractor_velocity=rv, name="noise_rms")],
        ["signal_rms", "noise_rms", "signal_rms/noise_rms"]
    ]
]


class TestTracewise:
    """Test tracewise QC metrics."""

    @pytest.mark.parametrize("metrics, names", METRICS_TO_COMPUTE)
    def test_qc_tracewise(self, survey, metrics, names):
        """Test `survey.qc_tracewise`."""
        _ = self
        survey.qc_tracewise(metrics=metrics, bar=False)
        for name in to_list(names):
            assert name in survey.qc_metrics
            header_cols = to_list(survey.qc_metrics[name].header_cols)
            assert all(col in survey.headers for col in header_cols)

    @pytest.mark.parametrize("metrics, metric_names", MMAP_METRICS)
    def test_construct_qc_map(self, survey, metrics, metric_names):
        """Test `survey.construct_qc_map`."""
        _ = self
        survey.qc_tracewise(metrics=metrics, bar=False)
        mmaps = survey.construct_qc_maps(by='shot', metric_names=metric_names)
        if metric_names is None:
            assert len(mmaps) == len(DEFAULT_TRACEWISE_METRICS)
            return

        if not isinstance(metric_names, list):
            assert not isinstance(mmaps, list)
        mmaps = to_list(mmaps)
        metric_names = to_list(metric_names)
        assert len(to_list(mmaps)) == len(to_list(metric_names))

        for mmap, name in zip(mmaps, metric_names):
            if "/" not in name:
                assert name in mmap.index_data

@pytest.mark.parametrize("header_index", ["TRACE_SEQUENCE_FILE", "CDP", ["CDP", "FieldRecord"]])
@pytest.mark.parametrize("inplace", [True, False])
class TestFilterMetrics:
    """Test survey filter by metric"""
    @pytest.mark.parametrize("metric, threshold", [[MaxClipsLen, None], [MaxConstLen, 2]])
    def test_filter_metrics(self, stat_segy, header_index, metric, threshold, inplace):
        """Check that `filter_by_metric` properly updates survey `headers`."""
        _ = self
        path, trace_data = stat_segy
        survey = Survey(path, header_index=header_index, header_cols="offset")

        traces_pos = survey.headers.reset_index()["TRACE_SEQUENCE_FILE"].values - 1
        sorted_ixs = np.argsort(traces_pos)
        trace_data = trace_data[sorted_ixs]
        gather = Gather(survey.headers.iloc[sorted_ixs], trace_data, survey.samples, survey)

        survey.qc_tracewise(metrics=metric, bar=False)
        survey_filtered = survey.filter_by_metric(metric_name=metric.name, threshold=threshold, inplace=inplace)

        metric_instance = survey.qc_metrics[metric.name]
        n_bad = metric_instance.binarize(metric_instance(gather), threshold=threshold).sum()

        assert gather.n_traces - n_bad == survey_filtered.headers.shape[0]
        assert_survey_processed_inplace(survey, survey_filtered, inplace)

    def test_remove_dead_traces(self, stat_segy, header_index, inplace):
        """Check that `remove_dead_traces` properly updates survey `headers`."""
        _ = self
        path, trace_data = stat_segy
        survey = Survey(path, header_index=header_index, header_cols="offset")

        traces_pos = survey.headers.reset_index()["TRACE_SEQUENCE_FILE"].values - 1
        trace_data = trace_data[np.argsort(traces_pos)]

        survey_copy = survey.copy()
        survey_filtered = survey.remove_dead_traces(inplace=inplace)

        is_dead = np.isclose(trace_data.min(axis=1), trace_data.max(axis=1))
        survey_copy.headers = survey_copy.headers.loc[~is_dead]
        survey_copy.headers["dead_trace"] = np.float32(0)

        # Validate that dead traces are not present
        assert survey_filtered.headers.index.is_monotonic_increasing
        assert_surveys_equal(survey_filtered, survey_copy)
        assert_survey_processed_inplace(survey, survey_filtered, inplace)
